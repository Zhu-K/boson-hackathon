# Flask Real-Time Streaming UI for Higgs Audio

## Overview
This Flask application provides a **real-time streaming** web interface for the Higgs Audio TTS emotion transformation system. Unlike Streamlit, this implementation uses WebSockets to stream audio chunks as they're generated, enabling true real-time playback.

## Key Features
- 🎭 **Dual Mode Support**: Choose between "Angry" and "Sarcastic" emotion modes
- 🎙️ **One-Click Recording**: Record audio with customizable duration (2-10 seconds)
- 🔍 **Live Transcription**: See your speech transcribed in real-time
- 🎭 **Emotion Translation**: Watch your text transform with emotion
- 🔊 **Real-Time Audio Streaming**: Audio starts playing immediately as chunks are generated
- 📡 **WebSocket Communication**: Bi-directional real-time communication
- 🎨 **Modern UI**: Beautiful gradient design with smooth animations

## How It Works

### Real-Time Streaming Architecture
1. **WebSocket Connection**: Client establishes a persistent WebSocket connection with the server
2. **Audio Generation**: Server generates TTS audio chunks using the Higgs Audio API
3. **Immediate Streaming**: Each audio chunk is sent to the client as soon as it's generated
4. **Web Audio API**: Client uses the Web Audio API to play chunks in sequence without gaps
5. **Seamless Playback**: Audio plays continuously as new chunks arrive

### Technical Stack
- **Backend**: Flask + Flask-SocketIO for WebSocket support
- **Frontend**: Vanilla JavaScript with Web Audio API
- **Communication**: Socket.IO for real-time bidirectional events
- **Audio Processing**: PCM audio chunks streamed and played in real-time

## Installation

1. Install the required dependencies:
```bash
pip install Flask==3.0.0 Flask-SocketIO==5.3.5 python-socketio==5.10.0
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your `.env` file contains the `BOSON_API_KEY`:
```
BOSON_API_KEY=your_api_key_here
```

## Running the App

Launch the Flask app with:
```bash
python flask_app.py
```

The app will start on `http://localhost:5000`

Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Select Mode**: Choose between "Angry" or "Sarcastic" emotion transformation
2. **Set Duration**: Adjust the slider for recording duration (2-10 seconds)
3. **Start Recording**: Click "🎙️ Start Recording" and speak into your microphone
4. **Real-Time Processing**:
   - Watch the status updates as your audio is processed
   - See the original transcription appear
   - View the emotionally transformed text
   - **Hear the audio start playing immediately** as it's generated
5. **Record Again**: Click the button again to start a new session

## Advantages Over Streamlit

### ✅ Real-Time Streaming
- Audio plays **as it's generated**, not after completion
- Significantly lower latency for user feedback
- True streaming experience

### ✅ Better Performance
- WebSocket connections are more efficient than HTTP polling
- Reduced server load with persistent connections
- Smoother user experience

### ✅ More Control
- Direct access to Web Audio API for precise audio playback
- Better handling of audio chunks and timing
- No buffering delays

## Technical Details

### Audio Streaming Process
1. Server receives TTS stream from Higgs Audio API
2. Each PCM audio chunk is base64-encoded
3. Chunk is immediately sent via WebSocket to client
4. Client decodes base64 to ArrayBuffer
5. PCM data is converted to Float32Array
6. AudioBuffer is created and scheduled for playback
7. Playback timing is managed to ensure seamless transitions

### Web Audio API
The client uses the Web Audio API to:
- Create an AudioContext for audio processing
- Convert PCM int16 data to float32 format
- Schedule audio buffers for gapless playback
- Maintain precise timing between chunks

## Troubleshooting

### Microphone Issues
- The app uses `input_device_index=1`. Adjust in `flask_app.py` if needed
- Check browser permissions for microphone access
- Verify system audio input settings

### WebSocket Connection Issues
- Ensure port 5000 is not blocked by firewall
- Check browser console for connection errors
- Try a different browser (Chrome/Firefox recommended)

### Audio Playback Issues
- Modern browsers require user interaction before playing audio
- Check browser audio permissions
- Verify system audio output is working
- Try refreshing the page if audio doesn't play

### API Errors
- Verify `BOSON_API_KEY` is set correctly in `.env`
- Ensure internet connectivity
- Check API rate limits

## Browser Compatibility
- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari (may require additional permissions)
- ⚠️ Older browsers may not support Web Audio API

## Performance Tips
- Use a stable internet connection for best streaming quality
- Close other audio applications to avoid conflicts
- Keep recording duration reasonable (5-7 seconds optimal)

## Security Note
The app runs on `0.0.0.0:5000` for development. For production:
- Use HTTPS for secure WebSocket connections (WSS)
- Implement proper authentication
- Add rate limiting
- Use a production WSGI server (e.g., Gunicorn with eventlet)
