## Spatial Vignettes Server ##

From project root run `python -m venv venv`, `source venv/bin/activate`

Install dependencies `pip install -r requirements.txt`

On Mac, download SAM2 models to `models` directory by running `brew install huggingface-cli` then `huggingface-cli download --local-dir models apple/coreml-sam2.1-baseplus`

Start server `SAM_BACKEND=coreml uvicorn main:app --reload`

Use ip address or ngrok. For ngrok: In new terminal, start ngrok `ngrok http 8000` (might need to set up account). Get the path in Forwarding.

Put the ip or ngrok path in capture app to connect to server.