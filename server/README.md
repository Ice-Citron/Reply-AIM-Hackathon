# AI Medical Assistant Chat Server

This directory contains backend server implementations for the AI Medical Assistant chat interface.

## Features

- 💬 Real-time chat with OpenAI GPT models
- 🏥 Medical-focused system prompts for appropriate responses
- 🔒 Secure API key handling via environment variables
- 🚨 Safety checks for emergency situations
- 📊 Token usage tracking
- ⚡ CORS enabled for frontend integration

## Available Server Implementations

### Node.js (Express)
- **File:** `chat-server.js`
- **Best for:** JavaScript/Node.js developers
- **Performance:** Excellent

### Python (Flask)
- **File:** `chat-server.py`
- **Best for:** Python developers
- **Performance:** Very good

## Quick Start

### Option 1: Node.js Server

1. **Install dependencies:**
   ```bash
   npm install express cors dotenv openai
   ```

2. **Create `.env` file:**
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Run the server:**
   ```bash
   node chat-server.js
   ```

4. **Access the chat:**
   - Open browser to: `http://localhost:3000/pages/chat.html`
   - API endpoint: `http://localhost:3000/api/chat`

### Option 2: Python Server

1. **Install dependencies:**
   ```bash
   pip install flask flask-cors openai python-dotenv
   ```

2. **Create `.env` file:**
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Run the server:**
   ```bash
   python chat-server.py
   ```

4. **Access the chat:**
   - Open browser to: `http://localhost:3000/pages/chat.html`
   - API endpoint: `http://localhost:3000/api/chat`

## Getting an OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key and add it to your `.env` file

## Environment Variables

Create a `.env` file in the `server` directory:

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-key-here

# Optional: Change the port (default: 3000)
PORT=3000

# Optional: OpenAI Model Selection
# OPENAI_MODEL=gpt-4          # More accurate, higher cost
# OPENAI_MODEL=gpt-3.5-turbo  # Faster, lower cost
```

## API Endpoints

### POST `/api/chat`
Send a chat message and receive AI response.

**Request:**
```json
{
  "message": "I have a headache, what should I do?",
  "history": [
    {
      "role": "user",
      "content": "Previous message"
    },
    {
      "role": "assistant",
      "content": "Previous response"
    }
  ]
}
```

**Response:**
```json
{
  "response": "I understand you're experiencing a headache...",
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350
  }
}
```

### GET `/api/health`
Check server health status.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Security Considerations

1. **Never commit your `.env` file** - It's already in `.gitignore`
2. **Use environment variables** for all sensitive data
3. **Implement rate limiting** in production
4. **Add authentication** for production use
5. **Monitor API usage** to control costs
6. **Validate all inputs** before sending to OpenAI

## Production Deployment

For production deployment, consider:

1. **Add authentication/authorization**
   - JWT tokens
   - API keys
   - OAuth

2. **Implement rate limiting**
   ```javascript
   // Example for Express
   const rateLimit = require('express-rate-limit');

   const limiter = rateLimit({
     windowMs: 15 * 60 * 1000, // 15 minutes
     max: 100 // limit each IP to 100 requests per windowMs
   });

   app.use('/api/', limiter);
   ```

3. **Add request logging**
4. **Set up error monitoring** (Sentry, etc.)
5. **Use a reverse proxy** (nginx)
6. **Enable HTTPS**
7. **Set up proper CORS** restrictions

## Cost Management

OpenAI charges based on token usage:

- **GPT-4:** ~$0.03 per 1K prompt tokens, ~$0.06 per 1K completion tokens
- **GPT-3.5-Turbo:** ~$0.001 per 1K tokens (much cheaper)

To reduce costs:
- Use `gpt-3.5-turbo` for general queries
- Implement conversation length limits
- Add caching for common questions
- Monitor usage via OpenAI dashboard

## Troubleshooting

### "Invalid API Key" Error
- Check that your `.env` file exists
- Verify the API key is correct
- Ensure no extra spaces in the `.env` file

### CORS Errors
- Make sure the server is running
- Check that CORS is enabled in the server code
- Verify the frontend is making requests to the correct URL

### Port Already in Use
- Change the PORT in `.env` file
- Kill the process using the port: `lsof -ti:3000 | xargs kill`

### OpenAI Rate Limits
- Implement exponential backoff
- Use a lower-tier model
- Add request queuing

## Testing

Test the API endpoint directly:

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are common symptoms of the flu?",
    "history": []
  }'
```

## License

This server implementation is provided as-is for the AI Medical Assistant project.

## Support

For issues or questions:
1. Check this README
2. Review the code comments
3. Check OpenAI API documentation
4. Open an issue in the repository
