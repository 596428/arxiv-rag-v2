// arXiv RAG Chat Edge Function
// POST /functions/v1/chat
import { serve } from 'https://deno.land/std@0.177.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { embedQuery } from './embedder.ts';
import { generateAnswer } from './generator.ts';
import type { ChatRequest, ChatResponse, Source, ChunkMatch } from './types.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  const totalStart = performance.now();

  try {
    // Parse request
    const { query, embedding_model = 'openai', history = [], top_k = 5 }: ChatRequest = await req.json();

    if (!query || query.trim().length === 0) {
      return new Response(
        JSON.stringify({ error: 'Query is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Get API keys from environment
    const openaiKey = Deno.env.get('OPENAI_API_KEY');
    const geminiKey = Deno.env.get('GEMINI_API_KEY');
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;

    if (!openaiKey) throw new Error('OPENAI_API_KEY not configured');
    if (!geminiKey) throw new Error('GEMINI_API_KEY not configured');

    // Initialize Supabase client
    const supabase = createClient(supabaseUrl, supabaseKey);

    // Step 1: Search for relevant chunks using direct SQL query (fast)
    const searchStart = performance.now();
    let embedTime = 0;

    // Use direct SQL query for text search
    const { data: chunks, error: searchError } = await supabase
      .from('chunks')
      .select('chunk_id, paper_id, content, section_title')
      .textSearch('content', query.split(' ').join(' | '), { type: 'websearch' })
      .limit(top_k);

    if (searchError) {
      throw new Error(`Search error: ${searchError.message}`);
    }

    const searchTime = Math.round(performance.now() - searchStart);

    // Format sources (map SQL column names to Source interface)
    const sources: Source[] = (chunks || []).map((chunk: any) => ({
      paper_id: chunk.paper_id,
      title: 'Research Paper',
      section: chunk.section_title || 'Unknown Section',
      similarity: chunk.rank ? Math.min(chunk.rank, 1) : 0.5,
      chunk_text: chunk.content?.substring(0, 500) || '',
    }));

    // Step 3: Generate answer with Gemini
    const { answer, time_ms: generateTime } = await generateAnswer(
      query,
      sources,
      history,
      geminiKey
    );

    const totalTime = Math.round(performance.now() - totalStart);

    // Build response
    const response: ChatResponse = {
      answer,
      sources,
      metrics: {
        embed_time_ms: embedTime,
        search_time_ms: searchTime,
        generate_time_ms: generateTime,
        total_time_ms: totalTime,
        chunks_found: sources.length,
        embedding_model,
      },
    };

    return new Response(JSON.stringify(response), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Chat error:', error);
    return new Response(
      JSON.stringify({ error: error.message || 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
