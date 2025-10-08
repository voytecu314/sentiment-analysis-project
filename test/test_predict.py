import pytest
from src.predict import load_model, predict_texts

# def test_predict_positive_sentence():
#     sentence = "I love this movie, it was fantastic and inspiring!"
#     model = load_model("models/sentiment.joblib")
#     prediction = predict_texts(model, [sentence])[0][0]
#     assert prediction == 1

# def test_predict_negative_sentence():
#     sentence = "The service was terrible and the food was awful."
#     model = load_model("models/sentiment.joblib")
#     prediction = predict_texts(model, [sentence])[0][0]
#     assert prediction == 0

@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("I love this movie, it was fantastic and inspiring!", 1),
        ("The service was terrible and the food was awful.", 0),
    ]
)
def test_predict_sentiment(sentence, expected):
    model = load_model("models/sentiment.joblib")
    prediction = predict_texts(model, [sentence])[0][0]
    assert prediction == expected