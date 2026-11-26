import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate.zac2026.guard import Guard, UnverifiedReference, _said
from lifemate.zac2026.vnnum import normalize_spoken


class GuardSourcesTest(unittest.TestCase):
    def guard(self, history=None):
        specs = {
            'smart_home.get_status': {'kind': 'read', 'parameters': {'properties': {
                'type': {'type': 'string', 'enum': ['light', 'fan']}, 'room': {'type': 'string'}}}},
            'alarm.create': {'parameters': {'required': ['time'], 'properties': {
                'time': {'type': 'string'}, 'label': {'type': 'string', 'x-match': 'fuzzy'}}}},
            'lights.set': {'parameters': {'required': ['device_id'], 'properties': {
                'device_id': {'type': 'string', 'x-ref': 'smart_home.devices'}}}},
        }
        row = {'context': {'current_time': '2026-11-12T10:00:00+07:00'}, 'history': history or []}
        return Guard(specs, row, {'alarm.create': {'label'}}, 'bật đèn phòng khách', current_text='bật đèn phòng khách')

    def test_vietnamese_enum_and_room(self):
        guard = self.guard()
        args = {'type': 'light', 'room': 'phòng khách'}
        self.assertEqual(guard.fix('smart_home.get_status', args), args)
        self.assertEqual(guard.decisions[0]['source'], 'current_user:device_type')

    def test_frequency_does_not_justify_invented_label(self):
        guard = self.guard()
        args = {'time': '2026-11-13T06:00:00+07:00', 'label': 'Đi làm'}
        self.assertEqual(guard.fix('alarm.create', args), {'time': args['time']})
        self.assertFalse(guard.decisions[-1]['kept'])

    def test_tool_source_scoped_to_collection(self):
        guard = self.guard()
        guard.observe('smart_home.get_status', {'ok': True, 'devices': [{'id': 'light_01'}]})
        guard.fix('lights.set', {'device_id': 'light_01'})
        self.assertEqual(guard.decisions[-1]['source'], 'tool_result:smart_home.get_status:devices')
        with self.assertRaisesRegex(UnverifiedReference, "device_id='rcp_01'.*ứng dụng smart_home"):
            guard.fix('lights.set', {'device_id': 'rcp_01'})
        self.assertFalse(guard.decisions[-1]['kept'])

    def test_failed_result_not_evidence(self):
        guard = self.guard()
        guard.observe('smart_home.get_status', {'ok': True, 'done': False, 'devices': [{'id': 'light_01'}]})
        with self.assertRaises(UnverifiedReference):
            guard.fix('lights.set', {'device_id': 'light_01'})

    def test_nested_or_wrong_collection_ids_do_not_supply_reference(self):
        guard = self.guard()
        guard.observe('lights.list', {'ok': True, 'devices': [{'id': 'light_01'}]})
        guard.observe('smart_home.get_status', {'ok': True, 'scenes': [{'id': 'light_01'}],
                                               'devices': [{'id': 'light_02', 'state': {'id': 'light_01'}}]})
        with self.assertRaises(UnverifiedReference):
            guard.fix('lights.set', {'device_id': 'light_01'})

    def test_optional_reference_is_blocked_before_it_can_be_dropped(self):
        guard = self.guard()
        guard.specs['lights.set']['parameters']['required'] = []
        with self.assertRaises(UnverifiedReference):
            guard.fix('lights.set', {'device_id': 'light_01'})
        self.assertEqual(guard.fix('lights.set', {}), {})

    def test_only_contact_fields_in_call_and_sms_records_supply_contact_ids(self):
        guard = self.guard()
        prop = {'x-ref': 'contacts.contacts'}
        guard.observe('phone.recent_calls', {'ok': True, 'calls': [
            {'id': 'call_123', 'contact': {'id': 'c_01'}}, {'id': 'call_124', 'contact_id': 'c_02'}]})
        guard.observe('sms.read_recent', {'ok': True, 'messages': [
            {'id': 'msg_456', 'contact_id': 'c_03', 'content': 'c_04'}]})
        for contact_id in ('c_01', 'c_02', 'c_03'):
            self.assertIsNotNone(guard.source('phone.call', 'contact_id', contact_id, prop))
        for other_id in ('call_123', 'msg_456', 'c_04'):
            self.assertIsNone(guard.source('phone.call', 'contact_id', other_id, prop))

    def test_singular_read_and_created_record_are_reference_evidence(self):
        row = {'context': {'current_time': '2026-11-12T10:00:00+07:00'}, 'history': []}
        specs = {f'{app}.cancel': {'parameters': {'properties': {
            'id': {'x-ref': ref}}}} for app, ref in [('ride_hailing', 'ride_hailing.rides'), ('alarm', 'alarm.alarms')]}
        guard = Guard(specs, row, {}, 'huỷ chuyến xe rồi xoá báo thức')
        guard.observe('ride_hailing.get_ride', {'ok': True, 'ride': {'id': 'ride_01'}})
        guard.record_result('alarm.create', {}, {'ok': True, 'id': 'alm_01', 'item': {'id': 'alm_01'}})
        self.assertEqual(guard.fix('ride_hailing.cancel', {'id': 'ride_01'}), {'id': 'ride_01'})
        self.assertEqual(guard.fix('alarm.cancel', {'id': 'alm_01'}), {'id': 'alm_01'})

    def test_scoped_historical_read_not_reused_as_unfiltered_read(self):
        guard = self.guard([{'tool_calls': [{'name': 'smart_home.get_status', 'arguments': {'room': 'phòng ngủ'}}],
                             'tool_results': [{'ok': True, 'devices': []}]}])
        self.assertIsNone(guard.cached('smart_home.get_status', {}))
        self.assertIsNotNone(guard.cached('smart_home.get_status', {'room': 'phòng ngủ'}))

    def test_evidence_snapshot_does_not_alias_result(self):
        guard = self.guard()
        result = {'ok': True, 'devices': [{'id': 'light_01'}]}
        guard.observe('smart_home.get_status', result)
        result['devices'].clear()
        guard.fix('lights.set', {'device_id': 'light_01'})
        self.assertTrue(guard.decisions[-1]['source'].startswith('tool_result:'))


class GuardOptionalTest(unittest.TestCase):
    specs = {
        'test.update': {'kind': 'write', 'parameters': {
            'required': ['target'],
            'properties': {
                'target': {'type': 'string'}, 'power': {'enum': ['on', 'off']},
                'limit': {'type': 'integer'}, 'ratio': {'type': 'number'}, 'date': {'format': 'date'},
                'due': {'format': 'date-time'}, 'query': {'type': 'string'}, 'items': {'type': 'array'},
                'settings': {'type': 'object'}, 'flag': {'type': 'boolean'}, 'seat': {'type': 'string'},
                'format_only': {'format': 'email'},
            },
        }},
    }
    row = {'context': {'current_time': '2026-11-12T10:00:00+07:00'}, 'history': []}

    def guard(self, text, whitelist=()):
        return Guard(self.specs, self.row, {'test.update': set(whitelist)}, text)

    def test_drops_unsupported_optionals_even_when_frequent(self):
        args = {'target': 'gốc', 'power': 'on', 'limit': 15, 'ratio': 0.5, 'date': '2026-11-13',
                'due': '2026-11-13T17:00:00+07:00', 'query': 'tự thêm', 'items': ['nhầm'],
                'settings': {'opt': True}, 'flag': True, 'format_only': 'x@example.com'}
        guard = self.guard('cập nhật mục gốc', args)
        self.assertEqual(guard.fix('test.update', args), {'target': 'gốc'})

    def test_keeps_explicit_numbers_and_entities_without_whitelist(self):
        guard = self.guard('chọn ghế a bốn và mười lăm vé, cho hai phẩy năm điểm')
        args = {'target': 'gốc', 'seat': 'A4', 'limit': 15, 'ratio': 2.5}
        self.assertEqual(guard.fix('test.update', args), args)

    def test_only_whitelisted_inference_with_matching_cue(self):
        args = {'power': 'on', 'date': '2026-11-13', 'due': '2026-11-13T17:00:00+07:00'}
        text = 'bật đèn lúc năm giờ chiều mai'
        self.assertEqual(self.guard(text).fix('test.update', args), {})
        self.assertEqual(self.guard(text, args).fix('test.update', args), args)
        wrong = {**args, 'power': 'off', 'date': '2026-11-14', 'due': '2026-11-13T19:00:00+07:00'}
        self.assertEqual(self.guard(text, args).fix('test.update', wrong), {})
        self.assertEqual(self.guard('không bật đèn', args).fix('test.update', args), {})
        self.assertEqual(self.guard('đặt ngay bây giờ chiều nay', {'due'}).fix('test.update', {
            'due': '2026-11-12T19:00:00+07:00'}), {})


class SpokenEvidenceTest(unittest.TestCase):
    def test_normalizes_vietnamese_count_time_and_code(self):
        samples = {
            'ghế a bốn': 'ghế A4', 'một trăm linh năm': '105', '1,2 triệu': '1200000',
            'hai triệu rưỡi': '2500000', 'mười lăm phẩy năm': '15.5', 'âm mười lăm': '-15',
            'không chín không một hai ba bốn năm sáu bảy': '0901234567',
            'chín giờ kém mười lăm tối mai': '20:45 tối mai',
            'chiều nay đặt ngay bây giờ': 'chiều nay đặt ngay bây giờ',
            'VTV sáu lúc 19h30': 'VTV6 lúc 19:30',
            'hỏi bà Tư, làm việc từ sau năm 2026': 'hỏi bà Tư, làm việc từ sau năm 2026',
            'một nghìn tỷ hai triệu': '1000002000000',
        }
        for original, expected in samples.items():
            with self.subTest(original=original):
                self.assertEqual(normalize_spoken(original), expected)

    def test_normalizes_both_sides_without_accepting_number_substrings(self):
        self.assertTrue(_said('ghế a bốn', 'Đổi sang ghế A4 nhé'))
        self.assertTrue(_said('A4', 'Đổi sang ghế a bốn nhé'))
        self.assertTrue(_said(15000, '15.000 đồng'))
        self.assertFalse(_said(15, '15.000 đồng'))
        self.assertFalse(_said(5, 'một phẩy năm'))
        self.assertFalse(_said(15, 'âm mười lăm'))
        self.assertFalse(_said(15, 'hẹn lúc 15:30'))
        self.assertFalse(_said('A4', 'Đổi sang ghế A45 nhé'))


if __name__ == '__main__':
    unittest.main()
