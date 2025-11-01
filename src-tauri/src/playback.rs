use anyhow::{Context, Result};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use std::{fs::File, io::BufReader, path::PathBuf, sync::mpsc, thread};

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
                        match File::open(&path)
                            .and_then(|f| Decoder::new(BufReader::new(f)).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
                        {
                            Ok(source) => {
                                match Sink::try_new(&handle) {
                                    Ok(s) => { s.append(source); s.play(); sink = Some(s); }
                                    Err(e) => eprintln!("audio: sink error: {e}"),
                                }
                            }
                            Err(e) => eprintln!("audio: file/decoder error for {}: {e}", path.display()),
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
