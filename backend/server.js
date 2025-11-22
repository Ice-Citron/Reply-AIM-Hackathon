import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Load hospital data
let hospitalsData = [];
const loadHospitalData = async () => {
  try {
    const data = await fs.readFile(
      path.join(__dirname, '../data/hospitals.json'),
      'utf-8'
    );
    hospitalsData = JSON.parse(data);
    console.log(`✅ Loaded ${hospitalsData.length} hospitals`);
  } catch (error) {
    console.error('❌ Error loading hospital data:', error.message);
  }
};

loadHospitalData();

// Tool: Search hospitals
function searchHospitals(args) {
  const {
    procedure_name,
    include_global = true,
    max_results = 10,
    budget_usd,
  } = args;

  console.log('🔍 Searching hospitals with args:', args);

  // Normalize procedure name
  const procedureKey = procedure_name
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/[^a-z_]/g, '');

  // Filter hospitals that offer this procedure
  let filtered = hospitalsData.filter((h) => {
    const hasProcedure = h.procedures && h.procedures[procedureKey];
    if (!hasProcedure) return false;

    // Filter by country if not including global
    if (!include_global && h.country !== 'US') return false;

    // Filter by budget
    if (budget_usd && h.procedures[procedureKey].price > budget_usd) {
      return false;
    }

    return true;
  });

  // Calculate scores and sort
  filtered = filtered.map((h) => {
    const price = h.procedures[procedureKey].price;
    const reliability = h.reliability_score;

    // Normalize price (inverse - lower is better)
    const maxPrice = Math.max(...filtered.map((x) => x.procedures[procedureKey].price));
    const minPrice = Math.min(...filtered.map((x) => x.procedures[procedureKey].price));
    const normalizedPrice = 1 - (price - minPrice) / (maxPrice - minPrice || 1);

    // Normalize reliability (higher is better)
    const normalizedReliability = reliability / 100;

    // Combined score: 40% price, 60% reliability
    const score = 0.4 * normalizedPrice + 0.6 * normalizedReliability;

    return {
      ...h,
      procedure_price: price,
      procedure_currency: h.procedures[procedureKey].currency,
      score,
    };
  });

  // Sort by score descending
  filtered.sort((a, b) => b.score - a.score);

  // Limit results
  filtered = filtered.slice(0, max_results);

  return {
    procedure: procedure_name,
    total_found: filtered.length,
    results: filtered.map((h) => ({
      hospital_id: h.id,
      name: h.name,
      country: h.country,
      city: h.city,
      state: h.state,
      lat: h.lat,
      lng: h.lng,
      reliability_score: h.reliability_score,
      procedure_price: h.procedure_price,
      currency: h.procedure_currency,
      score: Math.round(h.score * 100),
    })),
  };
}

// Tool: Get hospital details
function getHospitalDetails(args) {
  const { hospital_id } = args;
  const hospital = hospitalsData.find((h) => h.id === hospital_id);

  if (!hospital) {
    return { error: 'Hospital not found' };
  }

  return {
    ...hospital,
    all_procedures: Object.entries(hospital.procedures).map(
      ([name, details]) => ({
        procedure: name,
        ...details,
      })
    ),
  };
}

// System prompt for the AI
const SYSTEM_PROMPT = `You are "CareCompass", an AI consultant for US-based patients who want to understand and compare the COST and RELIABILITY of hospitals globally.

CRITICAL RULES:
- You DO NOT diagnose medical conditions or decide what treatment someone should have.
- You DO explain:
  - typical cost ranges for procedures and medications,
  - how prices differ between hospitals, cities, and countries,
  - how hospital reliability/quality is measured,
  - tradeoffs between staying in the US and traveling abroad (medical tourism).
- When suggesting hospitals or countries, rely ONLY on the structured data returned by tools.
- If data is missing or uncertain, say so clearly instead of guessing.
- Always encourage users to:
  - confirm all costs directly with the hospital or pharmacy,
  - discuss treatment decisions with a licensed clinician.

Safety guidelines:
- Do not tell users to ignore doctors or standard medical advice.
- Do not recommend specific off-label use of drugs.
- Do not recommend that users skip essential emergency care to save money.

Output style:
- Start with a short summary in 2–3 bullet points.
- Then show a simple comparison (markdown table) if multiple hospitals are involved.
- Use plain language, no jargon unless the user is clearly an expert.
- Always include reliability scores and prices in your comparisons.

Current procedures available in our database:
- hip_replacement
- knee_replacement
- heart_surgery

When users ask about symptoms or diagnoses, politely clarify that you can help them understand costs AFTER they have a diagnosis or procedure recommendation from a doctor.`;

// Tool definitions for OpenAI
const tools = [
  {
    type: 'function',
    function: {
      name: 'search_hospitals',
      description:
        'Find and rank hospitals by cost and reliability for a given procedure and location.',
      parameters: {
        type: 'object',
        properties: {
          procedure_name: {
            type: 'string',
            description:
              'Name of the medical procedure (e.g., "hip replacement", "knee replacement", "heart surgery")',
          },
          include_global: {
            type: 'boolean',
            description:
              'Include hospitals outside the US for medical tourism comparison',
            default: true,
          },
          max_results: {
            type: 'integer',
            description: 'Maximum number of hospitals to return',
            default: 10,
          },
          budget_usd: {
            type: 'number',
            description: 'Optional maximum budget in USD',
          },
        },
        required: ['procedure_name'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_hospital_details',
      description: 'Get detailed information about a specific hospital',
      parameters: {
        type: 'object',
        properties: {
          hospital_id: {
            type: 'string',
            description: 'Unique hospital ID',
          },
        },
        required: ['hospital_id'],
      },
    },
  },
];

// Main chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { messages } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages format' });
    }

    console.log('💬 Received chat request with', messages.length, 'messages');

    // Add system prompt to messages
    const messagesWithSystem = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...messages,
    ];

    // Call OpenAI
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: messagesWithSystem,
      tools: tools,
      tool_choice: 'auto',
    });

    let assistantMessage = response.choices[0].message;

    // Handle tool calls
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      console.log('🔧 AI requested tool calls:', assistantMessage.tool_calls.length);

      const toolMessages = [];

      for (const toolCall of assistantMessage.tool_calls) {
        const functionName = toolCall.function.name;
        const functionArgs = JSON.parse(toolCall.function.arguments);

        console.log(`  → Calling ${functionName} with:`, functionArgs);

        let result;
        if (functionName === 'search_hospitals') {
          result = searchHospitals(functionArgs);
        } else if (functionName === 'get_hospital_details') {
          result = getHospitalDetails(functionArgs);
        } else {
          result = { error: 'Unknown function' };
        }

        toolMessages.push({
          role: 'tool',
          tool_call_id: toolCall.id,
          name: functionName,
          content: JSON.stringify(result),
        });
      }

      // Second call to OpenAI with tool results
      const secondResponse = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          ...messagesWithSystem,
          assistantMessage,
          ...toolMessages,
        ],
      });

      assistantMessage = secondResponse.choices[0].message;
    }

    console.log('✅ Sending response');

    res.json({
      message: assistantMessage,
      usage: response.usage,
    });
  } catch (error) {
    console.error('❌ Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hospitals_loaded: hospitalsData.length,
  });
});

// Get all hospitals endpoint (for map view)
app.get('/api/hospitals', (req, res) => {
  res.json({
    hospitals: hospitalsData.map((h) => ({
      id: h.id,
      name: h.name,
      country: h.country,
      city: h.city,
      state: h.state,
      lat: h.lat,
      lng: h.lng,
      reliability_score: h.reliability_score,
      procedures: Object.keys(h.procedures),
    })),
  });
});

app.listen(PORT, () => {
  console.log(`\n🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health\n`);
});
