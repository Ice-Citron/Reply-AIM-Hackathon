# AI Medical Assistant Chat - Quick Start Guide

Welcome! This guide will help you get the chat interface running in just a few minutes.

## What You'll Need

- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Either Node.js (v18+) OR Python (3.8+)

## 🚀 Quick Start (Choose One)

### Option A: Node.js Server (Recommended)

1. **Navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up your API key:**
   ```bash
   cp .env.example .env
   # Then edit .env and add your OpenAI API key
   ```

4. **Start the server:**
   ```bash
   npm start
   ```

5. **Open the chat:**
   - Visit: `http://localhost:3000/pages/chat.html`
   - Start chatting with your AI Medical Assistant!

### Option B: Python Server

1. **Navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   ```bash
   cp .env.example .env
   # Then edit .env and add your OpenAI API key
   ```

4. **Start the server:**
   ```bash
   python chat-server.py
   ```

5. **Open the chat:**
   - Visit: `http://localhost:3000/pages/chat.html`
   - Start chatting with your AI Medical Assistant!

## 📁 File Structure

```
reply-aim-hackathon-ui/
├── webapp/
│   ├── pages/
│   │   ├── chat.html          # Chat interface (NEW!)
│   │   └── landing_page.html  # Landing page
│   └── css/
│       ├── main.css
│       └── tailwind.css
├── server/
│   ├── chat-server.js         # Node.js server
│   ├── chat-server.py         # Python server
│   ├── package.json
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
└── CHAT_QUICKSTART.md         # This file
```

## 🎨 Features

### Chat Interface Features:
- ✅ Real-time conversation with OpenAI GPT models
- ✅ Medical-focused responses with safety guidelines
- ✅ Conversation history maintenance
- ✅ Typing indicators
- ✅ Error handling
- ✅ Suggestion buttons for quick starts
- ✅ Responsive design matching the landing page
- ✅ Emergency situation detection

### Backend Features:
- ✅ Secure API key handling
- ✅ CORS enabled
- ✅ Error handling and validation
- ✅ Token usage tracking
- ✅ Health check endpoint
- ✅ Static file serving

## 🔑 Getting Your OpenAI API Key

1. Go to [https://platform.openai.com/](https://platform.openai.com/)
2. Sign up or log in
3. Click on your profile → "View API Keys"
4. Click "Create new secret key"
5. Copy the key (you won't be able to see it again!)
6. Paste it into your `.env` file

## 💰 Cost Information

The chat uses OpenAI's API which charges per token:

- **GPT-4:** ~$0.03-0.06 per 1K tokens (more accurate)
- **GPT-3.5-Turbo:** ~$0.001 per 1K tokens (faster, cheaper)

A typical conversation costs just a few cents. You can change the model in the server code.

## 🛠️ Configuration

Edit the `.env` file in the `server` directory:

```env
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-key-here

# Optional: Change server port (default: 3000)
PORT=3000

# Optional: Choose model (gpt-4 or gpt-3.5-turbo)
OPENAI_MODEL=gpt-4
```

## 🧪 Testing

Test the API directly:

```bash
# Health check
curl http://localhost:3000/api/health

# Send a chat message
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are symptoms of the flu?"}'
```

## ❗ Troubleshooting

### "Invalid API Key" Error
- Check your `.env` file exists in the `server` directory
- Verify the API key has no extra spaces
- Make sure you copied the entire key

### Port Already in Use
- Change `PORT` in `.env` to a different number (e.g., 3001)
- Or kill the process: `lsof -ti:3000 | xargs kill` (Mac/Linux)

### Chat Not Loading
- Make sure the server is running (check terminal for startup message)
- Check browser console for errors (F12)
- Verify you're accessing the correct URL

### CORS Errors
- Ensure the server has CORS enabled (it should by default)
- Try accessing via `http://localhost:3000` instead of `file://`

## 🔒 Security Notes

**IMPORTANT:**
- Never commit your `.env` file to Git
- Never share your OpenAI API key
- The `.env` file is already in `.gitignore`
- For production, add authentication and rate limiting

## 📚 Additional Documentation

- **Server README:** `server/README.md` - Detailed server documentation
- **OpenAI Docs:** [https://platform.openai.com/docs](https://platform.openai.com/docs)

## 🎯 Next Steps

1. Try asking the AI about common symptoms
2. Test different medical questions
3. Customize the system prompt in the server files
4. Add more features to the chat interface
5. Deploy to production (add authentication first!)

## 💡 Tips

- The AI has built-in safety checks for emergencies
- Conversation history is maintained for context
- You can clear the chat by refreshing the page
- The AI won't provide specific diagnoses (by design)
- For debugging, check browser console (F12) and server logs

## 🆘 Need Help?

1. Check the troubleshooting section above
2. Read `server/README.md` for detailed docs
3. Check browser console for errors
4. Check server terminal for error messages
5. Verify your OpenAI API key is valid

---

**Ready to chat?** Start the server and visit `http://localhost:3000/pages/chat.html`! 🏥
