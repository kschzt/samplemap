use anyhow::{bail, Context, Result};
use rodio::{buffer::SamplesBuffer, Decoder, OutputStream, Sink, Source};
use std::{fs::File, io::BufReader, path::PathBuf, sync::mpsc, thread};
use hound::{SampleFormat, WavReader};
use symphonia::core::{audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint};
use symphonia::default::get_probe;

pub enum Msg {
    Play(PathBuf),
    Stop,
}

#[derive(Clone)]
pub struct AudioHandle {
    tx: mpsc::Sender<Msg>,
}

impl AudioHandle {
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Msg>();
        thread::spawn(move || {
            // This thread owns the non-Send audio objects.
            let (_stream, handle) = match OutputStream::try_default() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("audio: failed to open output: {e}");
                    return;
                }
            };
            let mut sink: Option<Sink> = None;
            while let Ok(msg) = rx.recv() {
                match msg {
                    Msg::Stop => {
                        if let Some(s) = sink.take() { s.stop(); }
                    }
                    Msg::Play(path) => {
                        if let Some(s) = sink.take() { s.stop(); }
                        match decode_wav_to_source(&path) {
                            Ok(source) => match Sink::try_new(&handle) {
                                Ok(s) => { s.append(source); s.play(); sink = Some(s); }
                                Err(e) => eprintln!("audio: sink error: {e}"),
                            },
                            Err(e) => eprintln!("audio: wav decode error for {}: {e}", path.display()),
                        }
                    }
                }
            }
        });
        Ok(Self { tx })
    }

    pub fn play_path(&self, path: PathBuf) -> Result<()> { self.tx.send(Msg::Play(path)).context("send play") }
    pub fn stop(&self) { let _ = self.tx.send(Msg::Stop); }
}

fn decode_wav_to_source(path: &PathBuf) -> Result<SamplesBuffer<f32>> {
    // Try fast path (hound). If open fails, fall back to symphonia, then rodio.
    let mut reader = match WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            // Fallbacks when hound can't open this WAV
            if let Ok(buf) = decode_via_symphonia(path) { return Ok(buf); }
            // Final fallback: rodio generic decoder
            return decode_via_rodio(path).with_context(|| format!("open wav {} (hound) and symphonia failed: {}", path.display(), e));
        }
    };
    let spec = reader.spec();
    let channels = spec.channels as u16;
    let sample_rate = spec.sample_rate;

    // Collect and normalize to f32 [-1,1]
    let data: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|r| r.unwrap_or(0.0))
            .collect(),
        (SampleFormat::Float, 64) => {
            // hound doesn't expose f64 samples reliably; fall back to symphonia
            return decode_via_symphonia(path);
        }
        (SampleFormat::Int, 8) => reader
            .samples::<i8>()
            .map(|r| r.map(|v| (v as f32) / 128.0).unwrap_or(0.0))
            .collect(),
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|r| r.map(|v| (v as f32) / 32768.0).unwrap_or(0.0))
            .collect(),
        (SampleFormat::Int, 24) => {
            // Read as i32; assume 24-bit signed in lower bits, scale by 2^23.
            let scale = 8_388_608.0; // 2^23
            reader
                .samples::<i32>()
                .map(|r| r.map(|v| (v as f32) / scale).unwrap_or(0.0))
                .collect()
        }
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|r| r.map(|v| (v as f32) / 2_147_483_648.0).unwrap_or(0.0)) // 2^31
            .collect(),
        (fmt, bits) => {
            // Fallback path handles uncommon encodings (e.g., ADPCM, mu-law)
            if let Ok(buf) = decode_via_symphonia(path) { return Ok(buf); }
            return decode_via_rodio(path).with_context(|| format!("unsupported WAV format via hound {:?} {}-bit; symphonia fallback failed", fmt, bits));
        }
    };

    Ok(SamplesBuffer::new(channels, sample_rate, data))
}

fn decode_via_symphonia(path: &PathBuf) -> Result<SamplesBuffer<f32>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("wav");
    let probed = get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("probe wav {}", path.display()))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .cloned()
        .with_context(|| "no default audio track")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| "make decoder")?;

    let mut out: Vec<f32> = Vec::new();
    let mut out_spec = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::ResetRequired) => {
                // Stream parameters may have changed.
                let new_params = format
                    .default_track()
                    .ok_or_else(|| anyhow::anyhow!("no track after reset"))?
                    .codec_params
                    .clone();
                decoder = symphonia::default::get_codecs()
                    .make(&new_params, &DecoderOptions::default())?;
                continue;
            }
            Err(symphonia::core::errors::Error::IoError(_)) => break, // EOF
            Err(e) => return Err(e).context("read packet"),
        };

        if packet.track_id() != track.id { continue; }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::IoError(_)) => continue,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decode packet"),
        };

        let spec = *decoded.spec();
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        if out_spec.is_none() { out_spec = Some(spec); }
        out.extend_from_slice(buf.samples());
    }
    let spec = out_spec.ok_or_else(|| anyhow::anyhow!("empty audio"))?;
    Ok(SamplesBuffer::new(spec.channels.count() as u16, spec.rate, out))
}

fn decode_via_rodio(path: &PathBuf) -> Result<SamplesBuffer<f32>> {
    // As a last resort, let rodio decode and collect to f32 buffer.
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let dec = Decoder::new(BufReader::new(file)).context("rodio decode")?;
    let chans = dec.channels();
    let rate = dec.sample_rate();
    let data: Vec<f32> = dec.convert_samples::<f32>().collect();
    Ok(SamplesBuffer::new(chans, rate, data))
}
