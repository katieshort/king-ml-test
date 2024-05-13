import requests
import unittest


class TestAPI(unittest.TestCase):
    def setUp(self):
        # URL of the Flask API
        self.url = "http://127.0.0.1:5000/predict"

    def test_valid_request(self):
        """Test the API with valid input data."""
        data = [
            {
                "id": "33553",
                "c1": "VFc",
                "c2": "aW9z",
                "c3": "cHQ",
                "c4": "aHVhd",
                "c5": True,
                "c6": "LTAzOjAw",
                "n1": 303.488539,
                "n2": 1085.434206,
                "n3": 90.097357,
                "n4": 21.113713,
                "n5": 404.121039,
                "n6": 3859250336.874662,
                "n7": 10.312303,
                "n8": 1112.885066,
                "n10": 0.409991,
                "n11": 3.289646,
                "n12": 0.08629,
            }
        ]
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    def test_empty_request(self):
        """Test the API with an empty payload."""
        data = []
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 400)

    def test_incomplete_data(self):
        """Test the API with missing some optional fields."""
        data = [{"id": "33554", "c1": "VFc", "n1": 303.488539, "n2": 1085.434206}]
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    def test_with_extra_fields(self):
        """Test the API with extra fields that are not used."""
        data = [
            {
                "id": "33555",
                "c1": "VFc",
                "extra_field": "extra_value",
                "n1": 303.488539,
                "n2": 1085.434206,
            }
        ]
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", response.json())

    def test_invalid_id_type(self):
        """Test the API with an invalid type for the 'id' field."""
        data = [
            {
                "id": 33556,  # Should be a string, provided as int
                "c1": "VFc",
                "n1": 303.488539,
                "n2": 1085.434206,
            }
        ]
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn("ID must be a string", response.json()["error"])

    def test_missing_required_fields(self):
        """Test the API with missing required fields, such as 'id'."""
        # Data with the 'id' field missing
        data_missing_id = [
            {
                "c1": "VFc",
                "c2": "aW9z",
                "c3": "cHQ",
                "c4": "aHVhd",
                "c5": True,
                "c6": "LTAzOjAw",
                "n1": 303.488539,
                "n2": 1085.434206,
                "n3": 90.097357,
                "n4": 21.113713,
                "n5": 404.121039,
                "n6": 3859250336.874662,
                "n7": 10.312303,
                "n8": 1112.885066,
                "n10": 0.409991,
                "n11": 3.289646,
                "n12": 0.08629,
            }
        ]

        response = requests.post(self.url, json=data_missing_id)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required data fields", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
