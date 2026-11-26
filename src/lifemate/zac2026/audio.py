from pathlib import Path


class GipFormerASR:
    """Local GipFormer 1.5 INT8 transducer with CPU beam search."""
    def __init__(self, config):
        import sherpa_onnx

        path = Path(config['gipformer_path'])
        files = {name: path / f'{name}.int8.onnx' for name in ('encoder', 'decoder', 'joiner')}
        files['tokens'] = path / 'tokens.txt'
        for file in files.values():
            if not file.is_file():
                raise FileNotFoundError(f'Missing GipFormer model file: {file}')
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            **{key: str(value) for key, value in files.items()}, provider='cpu',
            num_threads=config['cpu_threads'], sample_rate=16000, feature_dim=80,
            decoding_method='modified_beam_search')

    def transcribe_batch(self, paths):
        import soundfile as sf

        streams = []
        for path in paths:
            samples, rate = sf.read(path, dtype='float32', always_2d=True)
            stream = self.recognizer.create_stream()
            stream.accept_waveform(rate, samples.mean(axis=1))
            streams.append(stream)
        self.recognizer.decode_streams(streams)
        return [stream.result.text.strip() for stream in streams]

    def close(self):
        self.recognizer = None
