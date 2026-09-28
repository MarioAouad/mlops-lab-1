import os

import requests
import streamlit as st

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000").rstrip("/")


def request_prediction(uploaded):
    response = requests.post(
        f"{INFERENCE_URL}/predict",
        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get("category"), str) or not 0 <= float(result["confidence"]) <= 1:
        raise ValueError("Inference returned an invalid prediction")
    return result


def main():
    st.title("Food-11 classifier")
    uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    st.image(uploaded, width=300)
    try:
        with st.spinner("Classifying image..."):
            result = request_prediction(uploaded)
    except requests.HTTPError as exc:
        st.error(f"Inference service returned {exc.response.status_code}: {exc.response.text}")
    except requests.RequestException:
        st.error("The inference service is unavailable. Please try again shortly.")
    except (ValueError, KeyError, TypeError):
        st.error("The inference service returned an invalid response.")
    else:
        st.write(f"**Prediction:** {result['category']} ({float(result['confidence']):.1%})")


if __name__ == "__main__":
    main()
