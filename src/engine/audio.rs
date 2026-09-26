use std::f32::consts::PI;

/// Decodes raw binary bytes from an uploaded audio container (WAV RIFF) into mono 16kHz f32 PCM samples.
/// If raw float samples are already provided, it validates and returns them.
pub fn decode_audio_to_pcm(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    // Check if it's a RIFF WAVE file
    if bytes.len() >= 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return parse_wav(bytes);
    }

    // If it's raw 16-bit PCM bytes (e.g. without header)
    if bytes.len().is_multiple_of(2) && bytes.len() >= 320 {
        let mut pcm = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.as_chunks::<2>().0 {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            pcm.push(sample as f32 / 32768.0);
        }
        return Ok(pcm);
    }

    Err("Unsupported audio format. Please provide a standard PCM WAV audio file.".to_string())
}

/// Lightweight, zero-dependency 16-bit / 32-bit RIFF WAV parser
fn parse_wav(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut offset = 12; // Skip "RIFF" (4) + size (4) + "WAVE" (4)
    let len = bytes.len();

    let mut channels: u16 = 1;
    let mut sample_rate: u32 = 16000;
    let mut bits_per_sample: u16 = 16;
    let mut audio_format: u16 = 1; // 1 = PCM, 3 = IEEE Float
    let mut pcm_data: Option<&[u8]> = None;

    while offset + 8 <= len {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        offset += 8;

        if chunk_id == b"fmt " && offset + chunk_size <= len {
            if chunk_size >= 16 {
                audio_format = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                channels = u16::from_le_bytes([bytes[offset + 2], bytes[offset + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
                bits_per_sample = u16::from_le_bytes([bytes[offset + 14], bytes[offset + 15]]);
            }
        } else if chunk_id == b"data" {
            let data_end = (offset + chunk_size).min(len);
            pcm_data = Some(&bytes[offset..data_end]);
            break;
        }

        offset += chunk_size;
    }

    let raw_data = pcm_data.ok_or_else(|| "WAV missing 'data' subchunk".to_string())?;
    let ch = channels as usize;
    if ch == 0 {
        return Err("Invalid WAV: 0 channels".to_string());
    }

    let mut mono_samples = Vec::new();

    if audio_format == 1 && bits_per_sample == 16 {
        // 16-bit PCM integer
        let bytes_per_frame = ch * 2;
        let num_frames = raw_data.len() / bytes_per_frame;
        mono_samples.reserve(num_frames);

        for f in 0..num_frames {
            let frame_offset = f * bytes_per_frame;
            let mut sum = 0.0f32;
            for c in 0..ch {
                let sample_offset = frame_offset + c * 2;
                let s = i16::from_le_bytes([raw_data[sample_offset], raw_data[sample_offset + 1]]);
                sum += s as f32 / 32768.0;
            }
            mono_samples.push(sum / ch as f32);
        }
    } else if audio_format == 3 && bits_per_sample == 32 {
        // 32-bit IEEE float
        let bytes_per_frame = ch * 4;
        let num_frames = raw_data.len() / bytes_per_frame;
        mono_samples.reserve(num_frames);

        for f in 0..num_frames {
            let frame_offset = f * bytes_per_frame;
            let mut sum = 0.0f32;
            for c in 0..ch {
                let sample_offset = frame_offset + c * 4;
                let s = f32::from_le_bytes([
                    raw_data[sample_offset],
                    raw_data[sample_offset + 1],
                    raw_data[sample_offset + 2],
                    raw_data[sample_offset + 3],
                ]);
                sum += s;
            }
            mono_samples.push(sum / ch as f32);
        }
    } else {
        return Err(format!(
            "Unsupported WAV format: format={}, bits={}",
            audio_format, bits_per_sample
        ));
    }

    // Resample to 16,000 Hz if necessary
    if sample_rate != 16000 && !mono_samples.is_empty() {
        return Ok(resample_linear(&mono_samples, sample_rate, 16000));
    }

    Ok(mono_samples)
}

/// Simple, high-speed linear resampler
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate == dst_rate {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let dst_len = (samples.len() as f64 / ratio).round() as usize;
    let mut resampled = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_idx = i as f64 * ratio;
        let base = src_idx.floor() as usize;
        let frac = (src_idx - base as f64) as f32;

        if base + 1 < samples.len() {
            let s0 = samples[base];
            let s1 = samples[base + 1];
            resampled.push(s0 + frac * (s1 - s0));
        } else if base < samples.len() {
            resampled.push(samples[base]);
        } else {
            break;
        }
    }

    resampled
}

/// Standard 80-channel Log-Mel Filterbank (Fbank) feature extractor for ASR models (e.g. SenseVoice)
/// Parameters: 16kHz, 25ms window (400 samples), 10ms frame shift (160 samples), 512 FFT, 80 Mel filters.
pub fn compute_fbank_80(samples: &[f32]) -> (Vec<f32>, usize) {
    let sample_rate = 16000.0f32;
    let frame_len = 400; // 25ms
    let frame_step = 160; // 10ms
    let n_fft = 512;
    let n_mels = 80;

    if samples.len() < frame_len {
        return (Vec::new(), 0);
    }

    let num_frames = (samples.len() - frame_len) / frame_step + 1;
    let mut fbank_features = Vec::with_capacity(num_frames * n_mels);

    // Precompute Hamming window
    let mut window = Vec::with_capacity(frame_len);
    for n in 0..frame_len {
        let w = 0.54 - 0.46 * ((2.0 * PI * n as f32) / (frame_len as f32 - 1.0)).cos();
        window.push(w);
    }

    // Precompute Mel filterbank matrix (80 x (n_fft / 2 + 1))
    let num_bins = n_fft / 2 + 1;
    let mel_filters = create_mel_filters(n_mels, num_bins, sample_rate);

    let mut frame_buf = vec![0.0f32; n_fft];
    let mut power_spectrum = vec![0.0f32; num_bins];

    for f in 0..num_frames {
        let start = f * frame_step;

        // Apply pre-emphasis and window
        for i in 0..frame_len {
            let sample = samples[start + i];
            frame_buf[i] = sample * window[i];
        }
        for i in frame_len..n_fft {
            frame_buf[i] = 0.0;
        }

        // Discrete Fourier Transform (real power spectrum)
        compute_power_spectrum(&frame_buf, &mut power_spectrum);

        // Apply 80 Mel filters
        for m in 0..n_mels {
            let filter = &mel_filters[m];
            let mut mel_energy = 0.0f32;
            for k in 0..num_bins {
                mel_energy += power_spectrum[k] * filter[k];
            }
            // Log-energy with floor to prevent negative infinity
            fbank_features.push((mel_energy.max(1e-5)).ln());
        }
    }

    (fbank_features, num_frames)
}

/// Compute power spectrum |X(k)|^2 for real signal using radix-2 Cooley-Tukey FFT.
/// Input length MUST be a power of 2 (e.g. 512).
fn compute_power_spectrum(signal: &[f32], output: &mut [f32]) {
    let n = signal.len();
    let num_bins = output.len();

    // Copy signal into real/imag arrays for in-place FFT
    let mut re = signal.to_vec();
    let mut im = vec![0.0f32; n];

    // In-place radix-2 FFT
    fft_in_place(&mut re, &mut im);

    // Compute power spectrum |X(k)|^2 / N
    let inv_n = 1.0 / n as f32;
    for k in 0..num_bins {
        output[k] = (re[k] * re[k] + im[k] * im[k]) * inv_n;
    }
}

/// In-place radix-2 Cooley-Tukey FFT. Input length must be a power of 2.
fn fft_in_place(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f32;
        let wn_re = angle.cos();
        let wn_im = angle.sin();

        let mut start = 0;
        while start < n {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;

            for k in 0..half {
                let even = start + k;
                let odd = start + k + half;

                let t_re = w_re * re[odd] - w_im * im[odd];
                let t_im = w_re * im[odd] + w_im * re[odd];

                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] += t_re;
                im[even] += t_im;

                let new_w_re = w_re * wn_re - w_im * wn_im;
                let new_w_im = w_re * wn_im + w_im * wn_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }

            start += len;
        }
        len <<= 1;
    }
}

/// Create triangular Mel filterbank matrix
fn create_mel_filters(n_mels: usize, num_bins: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let low_freq = 0.0f32;
    let high_freq = sample_rate / 2.0;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let low_mel = hz_to_mel(low_freq);
    let high_mel = hz_to_mel(high_freq);

    let mel_step = (high_mel - low_mel) / (n_mels + 1) as f32;
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        mel_points.push(low_mel + i as f32 * mel_step);
    }

    let mut bin_freqs = Vec::with_capacity(n_mels + 2);
    for mel in &mel_points {
        let hz = mel_to_hz(*mel);
        let bin = ((num_bins as f32 - 1.0) * hz / (sample_rate / 2.0)).round() as usize;
        bin_freqs.push(bin.min(num_bins - 1));
    }

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let mut filter = vec![0.0f32; num_bins];
        let left = bin_freqs[m];
        let center = bin_freqs[m + 1];
        let right = bin_freqs[m + 2];

        if center > left {
            for k in left..center {
                filter[k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for k in center..=right {
                filter[k] = (right - k) as f32 / (right - center) as f32;
            }
        }
        filters.push(filter);
    }

    filters
}

/// Whisper-style 128-channel log-Mel spectrogram, matching `WhisperFeatureExtractor` /
/// `whisper.audio.log_mel_spectrogram` exactly (used by Qwen3-ASR / Confucius4-R2T2):
/// 16kHz, n_fft=400, hop=160, Hann window, centered/reflect-padded STFT, Slaney-scale
/// mel filterbank (librosa `norm="slaney"`), then `log10 -> clamp(max-8) -> (x+4)/4`.
/// Input is always padded/truncated to exactly `n_samples` (30s @ 16kHz = 480_000), producing
/// exactly `nb_max_frames` (3000) output frames, since jetort's ONNX export uses a fixed-shape
/// 3000-frame audio encoder. Returns row-major `[n_mels, nb_max_frames]` (mel-major, matching
/// the `[1, 128, 3000]` tensor layout the encoder graph expects).
pub fn compute_log_mel_128(samples: &[f32]) -> Vec<f32> {
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;
    const N_MELS: usize = 128;
    const N_SAMPLES: usize = 480_000;
    const NB_MAX_FRAMES: usize = 3000;
    const NUM_BINS: usize = N_FFT / 2 + 1;

    // Pad/truncate to exactly 30s (right-padded with zero as `WhisperFeatureExtractor` does).
    let mut audio = vec![0.0f32; N_SAMPLES];
    let copy_len = samples.len().min(N_SAMPLES);
    audio[..copy_len].copy_from_slice(&samples[..copy_len]);

    // Center/reflect-pad by n_fft/2 on both sides (torch.stft(..., center=True) semantics).
    let pad = N_FFT / 2;
    let padded_len = audio.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    padded[pad..pad + audio.len()].copy_from_slice(&audio);
    for i in 0..pad {
        // reflect (excludes the edge sample itself, matching torch/numpy "reflect" mode)
        padded[pad - 1 - i] = audio[(i + 1).min(audio.len() - 1)];
        padded[pad + audio.len() + i] = audio[audio.len().saturating_sub(2 + i)];
    }

    // Hann window (periodic, matches torch.hann_window default).
    let mut window = [0.0f32; N_FFT];
    for n in 0..N_FFT {
        window[n] = 0.5 - 0.5 * ((2.0 * PI * n as f32) / N_FFT as f32).cos();
    }

    let mel_filters = create_mel_filters_slaney(N_MELS, NUM_BINS, 16000.0);

    let mut mel_spec = vec![0.0f32; N_MELS * NB_MAX_FRAMES];
    let mut frame = [0.0f32; N_FFT];
    let mut power = [0.0f32; NUM_BINS];
    // Precompute the DFT twiddle table once and reuse across all 3000 frames (n_fft=400 isn't a
    // power of 2, so a direct DFT is used instead of `fft_in_place`; recomputing cos/sin per
    // frame would be ~3000x wasted work since the table only depends on n_fft/num_bins).
    let twiddles = build_dft_twiddles(N_FFT, NUM_BINS);

    for t in 0..NB_MAX_FRAMES {
        let start = t * HOP_LENGTH;
        for i in 0..N_FFT {
            frame[i] = padded[start + i] * window[i];
        }
        compute_dft_power_spectrum(&frame, &twiddles, NUM_BINS, &mut power);
        for m in 0..N_MELS {
            let filter = &mel_filters[m];
            let mut e = 0.0f32;
            for k in 0..NUM_BINS {
                e += power[k] * filter[k];
            }
            mel_spec[m * NB_MAX_FRAMES + t] = e;
        }
    }

    let mut max_val = f32::NEG_INFINITY;
    for v in mel_spec.iter_mut() {
        *v = v.max(1e-10).log10();
        if *v > max_val {
            max_val = *v;
        }
    }
    let floor = max_val - 8.0;
    for v in mel_spec.iter_mut() {
        *v = (v.max(floor) + 4.0) / 4.0;
    }

    mel_spec
}

/// Precomputes `(cos, sin)` twiddle factors for a direct DFT: one `(cos, sin)` pair per
/// `(k, i)` combination, laid out row-major as `[num_bins][n_fft]`.
fn build_dft_twiddles(n_fft: usize, num_bins: usize) -> Vec<(f32, f32)> {
    let mut table = Vec::with_capacity(num_bins * n_fft);
    for k in 0..num_bins {
        let angle_step = -2.0 * PI * k as f32 / n_fft as f32;
        for i in 0..n_fft {
            let angle = angle_step * i as f32;
            table.push((angle.cos(), angle.sin()));
        }
    }
    table
}

/// Direct O(n_fft * num_bins) DFT magnitude^2 spectrum using a precomputed twiddle table
/// (see `build_dft_twiddles`). `n_fft=400` isn't a power of 2, so the radix-2 `fft_in_place`
/// used by `compute_fbank_80` doesn't apply here.
fn compute_dft_power_spectrum(signal: &[f32], twiddles: &[(f32, f32)], num_bins: usize, output: &mut [f32]) {
    let n = signal.len();
    for (k, out) in output.iter_mut().enumerate().take(num_bins) {
        let row = &twiddles[k * n..k * n + n];
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, &s) in signal.iter().enumerate() {
            let (c, sn) = row[i];
            re += s * c;
            im += s * sn;
        }
        *out = re * re + im * im;
    }
}

/// Slaney-scale (librosa `norm="slaney"`, `htk=False`) triangular Mel filterbank, matching
/// `whisper.audio.mel_filters` / `librosa.filters.mel`. Differs from `create_mel_filters`
/// (HTK-scale, used by the older SenseVoice fbank path) both in the mel-scale formula and in
/// per-filter area normalization.
fn create_mel_filters_slaney(n_mels: usize, num_bins: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let f_min = 0.0f32;
    let f_max = sample_rate / 2.0;

    let f_sp = 200.0f32 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    let hz_to_mel = |hz: f32| -> f32 {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        }
    };
    let mel_to_hz = |mel: f32| -> f32 {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        }
    };

    let min_mel = hz_to_mel(f_min);
    let max_mel = hz_to_mel(f_max);
    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let mel = min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32;
        mel_points.push(mel_to_hz(mel));
    }

    // FFT bin center frequencies
    let bin_freqs: Vec<f32> = (0..num_bins)
        .map(|k| k as f32 * sample_rate / ((num_bins - 1) as f32 * 2.0))
        .collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let left = mel_points[m];
        let center = mel_points[m + 1];
        let right = mel_points[m + 2];
        let mut filter = vec![0.0f32; num_bins];
        // Slaney-style area normalization: divide by the mel-band's own width in Hz.
        let enorm = 2.0 / (right - left);
        for (k, filt) in filter.iter_mut().enumerate() {
            let f = bin_freqs[k];
            let up = (f - left) / (center - left).max(1e-10);
            let down = (right - f) / (right - center).max(1e-10);
            let w = up.min(down).max(0.0);
            *filt = w * enorm;
        }
        filters.push(filter);
    }

    filters
}

/// Standard CTC Greedy Search Decoder:
/// Picks argmax for each frame, collapses consecutive duplicate tokens, and skips blank token (index 0).
pub fn ctc_greedy_decode(logprobs: &[f32], num_frames: usize, vocab_size: usize, blank_id: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    let mut prev_token = None;

    for t in 0..num_frames {
        let offset = t * vocab_size;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        for v in 0..vocab_size {
            let val = logprobs[offset + v];
            if val > max_val {
                max_val = val;
                max_idx = v;
            }
        }

        if max_idx != blank_id
            && prev_token != Some(max_idx) {
                tokens.push(max_idx as u32);
            }
        prev_token = Some(max_idx);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_synthetic_wav_bytes(num_samples: usize, sample_rate: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        let subchunk2_size = (num_samples * 2) as u32;
        let chunk_size = 36 + subchunk2_size;

        // RIFF header
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&chunk_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // "fmt " subchunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
        bytes.extend_from_slice(&1u16.to_le_bytes());  // PCM format
        bytes.extend_from_slice(&1u16.to_le_bytes());  // 1 channel (mono)
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());  // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // "data" subchunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&subchunk2_size.to_le_bytes());

        for i in 0..num_samples {
            let val = ((i as f32 * 0.1).sin() * 10000.0) as i16;
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        bytes
    }

    #[test]
    fn test_parse_wav_mono() {
        let wav_bytes = create_synthetic_wav_bytes(1600, 16000);
        let pcm = decode_audio_to_pcm(&wav_bytes).expect("Failed to parse WAV");
        assert_eq!(pcm.len(), 1600);
        assert!(pcm[0].abs() < 1.0);
    }

    #[test]
    fn test_fbank_80_computation() {
        // 800 samples = 50ms at 16kHz
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let (fbank, num_frames) = compute_fbank_80(&samples);
        assert!(num_frames > 0);
        assert_eq!(fbank.len(), num_frames * 80);
    }

    #[test]
    fn test_ctc_greedy_decode() {
        // Vocab size 5, 4 frames
        // Frame 0: token 1
        // Frame 1: token 1 (consecutive repeat -> collapsed)
        // Frame 2: token 0 (blank -> skipped)
        // Frame 3: token 2
        let vocab_size = 5;
        let mut logprobs = vec![-10.0f32; 4 * vocab_size];

        logprobs[0 * vocab_size + 1] = 5.0;
        logprobs[vocab_size + 1] = 5.0;
        logprobs[2 * vocab_size] = 5.0; // blank
        logprobs[3 * vocab_size + 2] = 5.0;

        let tokens = ctc_greedy_decode(&logprobs, 4, vocab_size, 0);
        assert_eq!(tokens, vec![1, 2]);
    }
}
