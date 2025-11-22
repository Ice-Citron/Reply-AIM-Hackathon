/**
 * Simple Express server for handling AI Medical Assistant chat requests
 * This server interfaces with OpenAI's API to provide medical information
 *
 * Setup:
 * 1. npm install express cors dotenv openai
 * 2. Create a .env file with: OPENAI_API_KEY=your_key_here
 * 3. Run: node chat-server.js
 */

const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('../webapp')); // Serve static files from webapp directory

// OpenAI Configuration
const OpenAI = require('openai');
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// System prompt for the medical assistant
const SYSTEM_PROMPT = `You are an AI Medical Assistant designed to provide general health information and guidance.

IMPORTANT GUIDELINES:
- Provide helpful, accurate medical information based on current medical knowledge
- Always remind users that this is general information and not a substitute for professional medical advice
- For serious symptoms or emergencies, always advise users to seek immediate medical attention
- Never provide specific diagnoses - only discuss possibilities and recommend seeing a healthcare provider
- Be empathetic and supportive
- Ask clarifying questions when needed to provide better guidance
- Remind users about medication interactions and the importance of consulting their doctor
- If asked about medications, provide general information but always recommend consulting a pharmacist or doctor
- For mental health concerns, be especially sensitive and recommend professional help when appropriate

EMERGENCY SITUATIONS:
If a user describes symptoms of a medical emergency (chest pain, difficulty breathing, severe bleeding, stroke symptoms, etc.), immediately advise them to call 911 or go to the emergency room.

Your goal is to be helpful while being responsible and ensuring users get proper medical care when needed.`;

// Chat endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { message, history = [] } = req.body;

        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Build messages array for OpenAI
        const messages = [
            { role: 'system', content: SYSTEM_PROMPT },
            ...history,
            { role: 'user', content: message }
        ];

        // Call OpenAI API
        const completion = await openai.chat.completions.create({
            model: 'gpt-4', // or 'gpt-3.5-turbo' for faster/cheaper responses
            messages: messages,
            max_tokens: 800,
            temperature: 0.7,
            presence_penalty: 0.1,
            frequency_penalty: 0.1
        });

        const aiResponse = completion.choices[0].message.content;

        // Return response
        res.json({
            response: aiResponse,
            usage: {
                prompt_tokens: completion.usage.prompt_tokens,
                completion_tokens: completion.usage.completion_tokens,
                total_tokens: completion.usage.total_tokens
            }
        });

    } catch (error) {
        console.error('OpenAI API Error:', error);

        if (error.status === 401) {
            res.status(401).json({ error: 'Invalid OpenAI API key. Please check your configuration.' });
        } else if (error.status === 429) {
            res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
        } else {
            res.status(500).json({ error: 'An error occurred processing your request. Please try again.' });
        }
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
    console.log(`🏥 AI Medical Assistant server running on port ${PORT}`);
    console.log(`📡 API endpoint: http://localhost:${PORT}/api/chat`);
    console.log(`🌐 Frontend: http://localhost:${PORT}/pages/chat.html`);

    if (!process.env.OPENAI_API_KEY) {
        console.warn('⚠️  WARNING: OPENAI_API_KEY not found in environment variables!');
        console.warn('   Please create a .env file with your OpenAI API key.');
    }
});
