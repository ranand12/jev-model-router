import unittest
from unittest.mock import patch
import httpx
import server

class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.body = {'model': 'auto', 'messages': [{'role': 'user', 'content': 'Hello'}]}

    def test_probability_policy_not_confidence(self):
        answer = {'choice': 'fast', 'probabilities': {'fast': .9, 'balanced': .08, 'deep': .02},
                  'confidence': 0}
        self.assertEqual(server.choose_tier(answer), ('fast', 'jev'))
        self.assertEqual(server.choose_tier(answer, True), ('balanced', 'tool-capability-floor'))

    def test_uncertain_and_missing_fallback(self):
        for answer in [None, [], {'choice': []}, {}, {'choice': 'fast', 'probabilities': {'fast': .34, 'balanced': .36, 'deep': .3}},
                       {'choice': 'fast', 'probabilities': {'fast': float('nan'), 'balanced': 0, 'deep': 0}}]:
            self.assertEqual(server.choose_tier(answer)[0], 'deep')

    @patch('server.call_vertex')
    @patch('server.ask_jev', side_effect=httpx.ReadTimeout('timeout'))
    def test_timeout_falls_back_and_keeps_tools(self, ask, vertex):
        tool_call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
        vertex.return_value = {'choices': [{'message': {'role': 'assistant', 'content': None,
                                                       'tool_calls': [tool_call]}}]}
        self.body['tools'] = [{'type': 'function', 'function': {'name': 'lookup'}}]
        response = self.client.post('/v1/chat/completions', json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Router-Tier'], 'deep')
        self.assertEqual(response.json['choices'][0]['message']['tool_calls'][0], tool_call)
        self.assertEqual(vertex.call_args.args[0]['tools'], self.body['tools'])

    @patch('server.call_vertex', return_value={'choices': [{'message': {'content': 'hello'}}]})
    @patch('server.ask_jev')
    def test_pinned_tier_skips_rerouting(self, ask, vertex):
        self.body['model'] = 'balanced'
        response = self.client.post('/v1/chat/completions', json=self.body)
        self.assertEqual(response.status_code, 200)
        ask.assert_not_called()
        self.assertEqual(vertex.call_args.args[1], 'balanced')

    def test_unsupported_requests_rejected(self):
        for extra in [{'stream': True}, {'model': 'arbitrary-paid-model'}, {'n': 20},
                      {'max_tokens': 100000}, {'messages': []}, {'model': []}]:
            response = self.client.post('/v1/chat/completions', json=self.body | extra)
            self.assertEqual(response.status_code, 400)

    @patch('server.google_access_token', return_value='test-token')
    @patch.dict('os.environ', {'GOOGLE_CLOUD_PROJECT': 'test-project'})
    @patch('server.httpx.post')
    def test_vertex_forwarding_preserves_history(self, post, token):
        body = self.body | {'messages': [
            {'role': 'assistant', 'tool_calls': [{'id': '1', 'type': 'function',
              'function': {'name': 'lookup', 'arguments': '{}'}, 'extra_content': {'signature': 'keep'}}]},
            {'role': 'tool', 'tool_call_id': '1', 'content': '{"status":"shipped"}'}]}
        post.return_value.json.return_value = {'choices': [{'message': {'content': 'done'}}]}
        server.call_vertex(body, 'deep')
        forwarded = post.call_args.kwargs['json']
        self.assertEqual(forwarded['messages'], body['messages'])
        self.assertEqual(forwarded['model'], server.MODELS['deep'])
        self.assertIn('/locations/global/endpoints/openapi/chat/completions', post.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
