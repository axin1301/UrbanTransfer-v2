import unittest
from urban_transfer.knowledge import parse_knowledge_json

class TestQwenKnowledgeParsing(unittest.TestCase):
    def test_markdown_fence(self):
        raw = '```json\n{"relevance":[0.1,0.8],"morphology":[0.2,0.3,0.4],"uncertainty":0.6,"explanation":"brief"}\n```'
        k = parse_knowledge_json(raw, 2)
        self.assertEqual(k.relevance, [0.1,0.8])

    def test_scalar_relevance_rejected(self):
        raw = '{"relevance":0.85,"morphology":[0.2,0.3,0.4],"uncertainty":0.6}'
        with self.assertRaisesRegex(ValueError, 'array'):
            parse_knowledge_json(raw, 2)

    def test_truncation_rejected(self):
        raw = '```json\n{"relevance":[0.1,0.8],"morphology":[0.2,0.3,0.4],"explanation":"truncated'
        with self.assertRaisesRegex(ValueError, 'complete JSON'):
            parse_knowledge_json(raw, 2)

    def test_wrong_number_sources_rejected(self):
        raw = '{"relevance":[0.3],"morphology":[0.2,0.3,0.4],"uncertainty":0.6}'
        with self.assertRaisesRegex(ValueError, 'Expected 2'):
            parse_knowledge_json(raw, 2)

if __name__ == '__main__':
    unittest.main()
