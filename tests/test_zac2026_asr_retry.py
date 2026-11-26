import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.zac2026.asr_retry import accepts_retry, needs_retry, pending_slot
from lifemate.zac2026.run import _transcribe_sample


class ASRRetryTest(unittest.TestCase):
    def test_short_reply_recovers_waiting_quantity(self):
        original = 'Ừ LẤY MÃ VÉ NHA EM'
        retry = 'Ừ LẤY BA VÉ NHA EM'
        history = [{'role': 'assistant', 'text': 'Chị cần mấy vé ạ?'}]
        slot = pending_slot(history)
        self.assertEqual(slot, 'quantity')
        self.assertTrue(needs_retry(slot, original, original))
        self.assertTrue(accepts_retry(slot, original, retry, retry))
        self.assertFalse(needs_retry(slot, retry, retry))
        self.assertFalse(needs_retry(slot, 'Dạ ba ạ', 'Dạ ba ạ'))

    def test_only_latest_unambiguous_question_counts(self):
        history = [{'role': 'assistant', 'text': 'Bạn cần mấy vé?'},
                   {'role': 'user', 'audio': 'turn.wav'}]
        self.assertIsNone(pending_slot(history))
        self.assertIsNone(pending_slot([{'role': 'assistant', 'text': 'Cần mấy vé và đi lúc mấy giờ?'}]))
        self.assertIsNone(pending_slot([{'role': 'assistant', 'text': 'Mình còn mấy ngày để làm ạ.'}]))

    def test_structured_question_and_valid_times(self):
        turn = {'role': 'assistant', 'text': 'Mình bổ sung giúp em ạ.', 'tool_calls': [
            {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'alarm.create', 'params': ['time']}}]}
        self.assertEqual(pending_slot([turn]), 'time')
        self.assertFalse(needs_retry('time', 'sáu giờ rưỡi chiều', '18:30 chiều'))
        self.assertFalse(accepts_retry('time', 'tối em', '99:90 tối em', '99:90 tối em'))

    def test_does_not_retry_cancellations_or_accept_unrelated_decodes(self):
        self.assertFalse(needs_retry('quantity', 'à thôi khỏi đặt vé nữa', 'à thôi khỏi đặt vé nữa'))
        self.assertFalse(accepts_retry('quantity', 'lấy mã vé', 'bốn hộp cam và hai suất cơm', 'bốn hộp cam và hai suất cơm'))
        self.assertFalse(accepts_retry('quantity', 'lấy mã vé', 'lấy vé', 'lấy vé'))

    def test_pipeline_preserves_raw_audio_and_retries_only_current_clip(self):
        rows = [{'id': 'sample', 'audio': 'request.wav', 'history': [
            {'role': 'user', 'audio': 'history.wav'}, {'role': 'assistant', 'text': 'Chị lấy mấy vé?'}]}]
        root = Path('split')
        original, retry = 'Ừ lấy mã vé nha em', 'Ừ lấy ba vé nha em'
        asr = Mock()
        asr.transcribe_batch.side_effect = [[original, 'đặt vé xem phim'], [retry]]
        result = _transcribe_sample(asr, root, rows[0])
        self.assertEqual(asr.transcribe_batch.call_args_list[1].args, ([root / 'request.wav'],))
        self.assertEqual(result, {'history.wav': 'đặt vé xem phim', 'request.wav': retry})


if __name__ == '__main__':
    unittest.main()
