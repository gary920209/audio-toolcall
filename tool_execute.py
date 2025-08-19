import asyncio
import json
import re
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from speechtool import SpeechCopilotTool

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"desta_tool_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

class DestaTool:
    """DeSTA model with tool calling capabilities"""
    
    def __init__(self, desta_model=None, openai_api_key: str = None):
        """Initialize DeSTA with tool integration"""
        # Initialize DeSTA model
        self.desta_model = desta_model
        if self.desta_model is None:
            try:
                from desta import DeSTA25AudioModel
                logger.info("Loading DeSTA model...")
                self.desta_model = DeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
                self.desta_model.to("cuda")
                self.desta_model.eval()
                logger.info("DeSTA model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DeSTA model: {e}")
                self.desta_model = None
        
        # Initialize tools
        self.tool = SpeechCopilotTool(openai_api_key=openai_api_key)
        
        # Tool function descriptions
        self.tool_descriptions = """
Available SpeechCopilotTool Functions:
SPEECH ANALYSIS:
- speaker_identification(audio_path): Identify and extract speaker embeddings using pyannote/embedding model. Returns speaker_id based on vocal characteristics and feature hash.
- speech_recognition(audio_path): Convert speech to text using local Whisper model. Supports multiple languages with automatic detection. Returns transcribed text with confidence scores.
- emotion_recognition(audio_path): Detect emotional state in speech using emotion2vec model. Analyzes prosodic features to classify emotions (happy, sad, angry, neutral, etc.) with confidence scores.
SOUND ANALYSIS:
- sound_classification(audio_path): Classify environmental sounds and audio events using Audio Spectrogram Transformer (AST). Detects sounds like animal noises, mechanical sounds, natural sounds, etc.
- sound_event_detection(audio_path): Detect and temporally locate sound events in audio using DeSTA audio reasoning model. Identifies what sounds occur and when.
- sound_duration_analysis(audio_path): Analyze temporal characteristics including total duration, active segments, silence periods, energy distribution, and spectral activity patterns.
MUSIC ANALYSIS:
- melody_recognition(audio_path): Extract melodic content using pitch tracking. Returns pitch sequences, note sequences, pitch range, and melodic contour analysis.
- chord_recognition(audio_path): Identify chords in music using chroma features and template matching. Detects major/minor triads across all 12 pitch classes with timestamps.
- harmonic_analysis(audio_path): Analyze harmonic structure including key detection, harmonic complexity, and harmonic-to-percussive ratio using chroma and tonnetz features.
- harmonic_tension_analysis(audio_path): Measure harmonic tension and dissonance levels. Identifies tension points based on dissonant intervals (minor 2nd, tritone, minor 7th, major 7th).
- harmonic_function_analysis(audio_path): Classify chords by harmonic function (tonic, subdominant, dominant) in tonal music context. Assumes C major key for analysis.
- harmonic_role_analysis(audio_path): Categorize harmonies by their musical role (structural, transitional, coloristic) and analyze harmonic rhythm patterns.
- dominant_chord_analysis(audio_path): Specifically analyze dominant chords (G, G7, D7, A7, etc.) and their resolutions in harmonic progressions.
- time_signature_analysis(audio_path): Detect time signature (4/4, 3/4, 2/4, etc.) using beat tracking and onset analysis. Returns beats per minute and beat consistency.
- rhythm_analysis(audio_path): Analyze rhythmic complexity including syncopation level, beat strength, rhythm variability, and onset patterns.
- chord_progression_recognition(audio_path): Identify common chord progressions (I-V-vi-IV, ii-V-I, etc.) and convert to Roman numeral analysis.
- chord_duration_analysis(audio_path): Measure how long each chord lasts in the progression and calculate average chord durations.
- instrument_recognition(audio_path): Identify musical instruments present in audio using specialized instrument detection models.
- genre_analysis(audio_path): Classify music genre using audio features and machine learning models trained on diverse music datasets.
**If no suitable function, you should answer the question directly based on the audio content.**
"""
    def openai_tool_call(self, audio_path: str, question: str) -> Dict[str, Any]:
        """Process a question with GPT and execute tool calls if needed"""
        import openai
        openai.api_key = self.tool.openai_api_key
        # Step 1: First GPT inference - decide what tools are needed
        logger.info("🤖 Step 1: GPT analyzing problem and deciding on tool usage...")
        initial_messages = [
            {
                "role": "system",
                "content": """Focus on the instructions. choose the tools you need to solve the problem, you can use more than one tools to solve the problems.
                If you need tools, respond like:
                    emotion_recognition("path")
                    speaker_identification("path")
                """
            },
            {   "role": "user", 
                "content": f"Question: {question}\n\n{self.tool_descriptions}\n\n Provide several tool calls if needed.",
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=initial_messages,
                max_tokens=512,
                temperature=0.3,
                top_p=1.0,
                n=1
            )

            gpt_response = response.choices[0].message.content.strip()
            logger.info(f"GPT initial response: {gpt_response}")
            function_calls = self.parse_tool_calls(gpt_response)
            if not function_calls:
                # No tools needed, return direct answer
                logger.info("✅ GPT provided direct answer (no tools needed)")
                return {
                    'type': 'direct_answer',
                    'response': gpt_response,
                    'tool_calls': [],
                    'tool_results': {},
                    'final_response': gpt_response
                }
            # Step 2: Execute tools
            logger.info(f"🔧 Step 2: Executing {len(function_calls)} tool(s)...")
            tool_results = self.execute_tools(function_calls, audio_path)
            formatted_results = self.format_tool_results(tool_results)
            logger.info(f"📊 Tool execution results: {formatted_results}")
            final_messages = [
                {
                    "role": "system",
                    "content": "Based on the tool execution results, provide a comprehensive answer to the user's question. Be specific and cite the relevant findings from the analysis."
                },
                {
                    "role": "user",
                    "content": f"Original question: {question}\n\nTool results:\n{formatted_results}\n\nProvide a complete answer based on these analysis results."
                }
            ]
                
            final_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=final_messages,
                temperature=0.1,
                max_tokens=800
            )
            
            final_answer = final_response.choices[0].message.content.strip()
            
            return {
                'type': 'tool_assisted',
                'initial_response': gpt_response,
                'tool_calls': function_calls,
                'tool_results': tool_results,
                'formatted_results': formatted_results,
                'final_response': final_answer
            }
        except Exception as e:
            logger.error(f"Error during OpenAI tool call: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
              
    def parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse function calls from DeSTA output"""
        function_calls = []
        
        # Handle both string and list inputs
        if isinstance(text, list):
            text = text[0] if text else ""
        
        # Extract text content if it's an object with .text attribute
        if hasattr(text, 'text'):
            text = text.text
        
        text = str(text).strip()
        lines = text.split('\n')
        patterns = [
            r'(\w+)\s*\(\s*["\']?([^"\')]+)["\']?\s*\)',  # function("arg")
            r'(\w+)\s*\(\s*audio_path\s*=\s*["\']?([^"\']+)["\']?\s*\)',  # function(audio_path="...")
            r'(\w+)\s*\(\s*\)',  # function()
            r'(\w+)\s*\(\s*["\']?(path)["\']?\s*\)',  # function("path")
            r'(\w+)\s*\((.*?)\)',  # generic: function(arg1, arg2, ...)
        ]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            matched = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    func_name = match.group(1)
                    args_str = match.group(2) if match.lastindex >= 2 else ""

                    # Split multiple arguments if necessary
                    args = [arg.strip().strip('"\'') for arg in args_str.split(',')] if args_str else []

                    # Replace placeholders like path/audio_path/<AUDIO>
                    args = ['path' if arg in ['path', 'audio_path', '<AUDIO>', '<audio>','<path_to_audio>','path_to_audio','audio_file','path_to_audio_sample'] else arg for arg in args]

                    function_calls.append({
                        'function': func_name,
                        'args': args,
                        'raw_call': line,
                    })
                    matched = True
                    break
            if not matched and ('(' in line and ')' in line):
                # Fallback: try to extract function name manually
                func_match = re.match(r'(\w+)\s*\(', line)
                if func_match:
                    func_name = func_match.group(1)
                    function_calls.append({
                        'function': func_name,
                        'args': ['path'],
                        'raw_call': line
                    })
                    logger.debug(f"Fallback parsing for: {line}")
                    
        return function_calls

    def _parse_arguments(self, args_str: str) -> List[Any]:
        """Parse function arguments safely"""
        args = []
        
        if ',' in args_str:
            # Multiple arguments
            parts = args_str.split(',')
            for part in parts:
                part = part.strip()
                args.append(self._parse_single_arg(part))
        else:
            # Single argument
            args.append(self._parse_single_arg(args_str))
            
        return args

    def _parse_single_arg(self, arg: str) -> Any:
        """Parse single argument"""
        arg = arg.strip()
        
        # Remove quotes if present
        if (arg.startswith('"') and arg.endswith('"')) or \
           (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]
            
        # Handle nested function calls
        if '(' in arg and ')' in arg:
            return arg
            
        return arg

    async def execute_tools(self, function_calls: List[Dict], audio_path: str) -> Dict[str, Any]:
        """Execute the parsed function calls"""
        results = {}
        
        for i, call in enumerate(function_calls):
            func_name = call['function']
            args = call['args']
            raw_call = call['raw_call']
            
            logger.info(f"🔧 Executing: {raw_call}")
            
            try:
                # Replace 'path' placeholder with actual audio path
                processed_args = []
                for arg in args:
                    if isinstance(arg, str) and arg == 'path':
                        processed_args.append(audio_path)
                    elif isinstance(arg, str) and '(' in arg:
                        # Handle nested function calls (context references)
                        for prev_func in results:
                            if prev_func in arg:
                                processed_args.append(results[prev_func])
                                break
                        else:
                            processed_args.append(arg)
                    else:
                        processed_args.append(arg)
                
                # Execute the function
                if hasattr(self.tool, func_name):
                    func = getattr(self.tool, func_name)
                    
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*processed_args)
                    else:
                        result = func(*processed_args)
                    
                    # Store result
                    result_key = f"{func_name}_{i}"
                    results[result_key] = result
                    results[func_name] = result  # Also store with function name for easy reference
                    
                    logger.info(f"✅ {func_name} completed")
                    
                else:
                    error_result = {"error": f"Function {func_name} not found"}
                    results[f"{func_name}_{i}"] = error_result
                    logger.warning(f"❌ {func_name} not found")
                    
            except Exception as e:
                error_result = {"error": str(e)}
                results[f"{func_name}_{i}"] = error_result
                logger.error(f"❌ {func_name} failed: {e}")
        
        return results

    def format_tool_results(self, results: Dict[str, Any]) -> str:
        """Format tool execution results for DeSTA"""
        formatted_results = []
        
        for key, result in results.items():
            # Skip duplicate entries (only process indexed ones)
            if '_' in key and key.split('_')[-1].isdigit():
                if isinstance(result, dict):
                    if 'error' in result:
                        formatted_results.append(f"{key}: Error - {result['error']}")
                    else:
                        # Extract key information based on function type
                        if 'text' in result:  # speech_recognition
                            formatted_results.append(f"{key}: Recognized text - \"{result['text']}\"")
                        elif 'primary_emotion' in result:  # emotion_recognition
                            formatted_results.append(f"{key}: Emotion - {result['primary_emotion']} (confidence: {result.get('confidence', 0):.2f})")
                        elif 'speaker_id' in result:  # speaker_identification
                            formatted_results.append(f"{key}: Speaker ID - {result['speaker_id']}")
                        elif 'classification' in result:  # sound_classification
                            formatted_results.append(f"{key}: Sound type - {result['classification']} (confidence: {result.get('confidence', 0):.2f})")
                        elif 'chords' in result:  # chord_recognition
                            formatted_results.append(f"{key}: Detected chords - {', '.join(result['chords'][:5])}")
                        elif 'genre' in result:  # genre_analysis
                            formatted_results.append(f"{key}: Music genre - {result['genre']} (confidence: {result.get('confidence', 0):.2f})")
                        elif 'key' in result:  # harmonic_analysis
                            formatted_results.append(f"{key}: Musical key - {result['key']}")
                        elif 'tempo' in result:  # rhythm_analysis
                            formatted_results.append(f"{key}: Tempo - {result['tempo']:.1f} BPM")
                        elif 'response' in result:  # query_LLM
                            formatted_results.append(f"{key}: AI analysis - {result['response']}")
                        else:
                            # Generic formatting
                            key_info = []
                            for k, v in result.items():
                                if k not in ['timestamp', 'function'] and not k.startswith('_'):
                                    if isinstance(v, (int, float, str)) and len(str(v)) < 50:
                                        key_info.append(f"{k}: {v}")
                            if key_info:
                                formatted_results.append(f"{key}: {', '.join(key_info[:3])}")
        
        return '\n'.join(formatted_results)

    async def process_query(self, audio_path: str, question: str) -> Dict:
        """Complete workflow: DeSTA decides -> Tool execution -> DeSTA reasoning"""
        
        # Step 1: First DeSTA inference - decide if tools are needed
        logger.info("🤖 Step 1: DeSTA analyzing audio and deciding on tool usage...")

        initial_messages = [
            {
                "role": "system",
                "content": """Focus on the audio clips and instructions. You have two options: If you can answer the question directly, put your answer in the format \"The correct answer is: \"___\" \"

Or, if you need additional analysis tools, respond with ONLY Python function calls (one per line) using the available tools, you can use more than one tools to solve the problems.

For tool calls, respond ONLY with function calls like:
melody_recognition("path")
use "path" to refer to the audio file, do not use any other words or explanations, just the function calls.

"""
            },
            {
                "role": "user", 
                "content": f"<|AUDIO|>\n\nQuestion: {question}\n\n{self.tool_descriptions}\n\nEither answer directly or provide several tool calls if needed.",
                "audios": [{
                    "audio": audio_path,
                    "text": f"Question: {question}\n\n{self.tool_descriptions}"
                }]
            }
        ]
        
        # Generate initial response
        initial_output = self.desta_model.generate(
            messages=initial_messages,
            do_sample=False,
            top_p=1.0,
            temperature=0.3,
            max_new_tokens=512
        )
        
        # Extract response text
        if hasattr(initial_output, 'text'):
            initial_response = initial_output.text
        elif isinstance(initial_output, list) and len(initial_output) > 0:
            initial_response = initial_output[0] if isinstance(initial_output[0], str) else str(initial_output[0])
        else:
            initial_response = str(initial_output)
        
        logger.info(f"DeSTA initial response: {initial_response}")
        
        # Step 2: Check if tools are needed
        function_calls = self.parse_tool_calls(initial_response)
        
        if not function_calls:
            # No tools needed, return direct answer
            logger.info("✅ DeSTA provided direct answer (no tools needed)")
            return {
                'type': 'direct_answer',
                'response': initial_response,
                'tool_calls': [],
                'tool_results': {},
                'final_response': initial_response
            }
        
        # Step 3: Execute tools
        logger.info(f"🔧 Step 2: Executing {len(function_calls)} tool(s)...")
        tool_results = await self.execute_tools(function_calls, audio_path)
        formatted_results = self.format_tool_results(tool_results)
        
        logger.info("📊 Tool execution results:")
        logger.info(formatted_results)
        
        # Step 4: Second DeSTA inference with tool results
        logger.info("🤖 Step 3: DeSTA analyzing tool results and providing final answer...")
        
        final_messages = [
            {
                "role": "system",
                "content": "You are an expert audio analysis assistant. Based on the tool execution results provided, answer the user's original question comprehensively and accurately."
            },
            {
                "role": "user",
                "content": f"Original question: {question}"
            },
            {
                "role": "assistant", 
                "content": initial_response
            },
            {
                "role": "user",
                "content": f"Tool execution results:\n{formatted_results}\n\nBased on these results, please provide a comprehensive answer to the original question: \"{question}\", the answer should be in the format like 'the correct answer is: <your answer here>'. Do not rule out the choices."
            }
        ]
        
        # Generate final response
        final_output = self.desta_model.generate(
            messages=final_messages,
            do_sample=False,
            top_p=1.0,
            temperature=0.3,
            max_new_tokens=512
        )
        
        # Extract final response
        if hasattr(final_output, 'text'):
            final_response = final_output.text
        elif isinstance(final_output, list) and len(final_output) > 0:
            final_response = final_output[0] if isinstance(final_output[0], str) else str(final_output[0])
        else:
            final_response = str(final_output)
        
        logger.info(f"🎯 Final answer: {final_response}")
        
        return {
            'type': 'tool_assisted',
            'initial_response': initial_response,
            'tool_calls': function_calls,
            'tool_results': tool_results,
            'formatted_results': formatted_results,
            'final_response': final_response
        }

# Example usage
async def main():
    """Process all questions from mmau-test-mini.json and save results"""
    from dotenv import load_dotenv
    load_dotenv()
    OPEN_API_KEY = os.getenv("OPEN_API_KEY")
    # Initialize the system
    desta_tool = DestaTool(
        openai_api_key=OPEN_API_KEY
    )
    
    logger.info("🎵 DeSTA Processing All Questions from mmau-test-mini.json")
    logger.info("=" * 60)
    
    # Load questions from JSON file
    try:
        with open("mmau-test-mini.json", "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        logger.info(f"Loaded {len(questions_data)} questions from mmau-test-mini.json")
    except Exception as e:
        logger.error(f"❌ Error loading JSON file: {e}")
        return
    
    # Process all questions
    results = []
    
    for i, item in enumerate(questions_data):
        logger.info(f"\n🔍 Processing question {i+1}/{len(questions_data)}")
        logger.info(f"ID: {item.get('id', 'N/A')}")
        logger.info(f"Audio: {item.get('audio_id', 'N/A')}")
        logger.info(f"Question: {item.get('question', '')[:100]}...")
        logger.info("-" * 50)
        
        try:
            # Extract data
            audio_id = item.get("audio_id", "")
            question = item.get("question", "")
            choices = item.get("choices", [])
            question += "Choose from the following options: "
            # use "or" for last option
            for i, option in enumerate(choices):
                question += f"\"{option}\""
                if i == len(choices) - 2:
                    question += " or "
                else:
                    question += ", "
            # Process with DeSTA
            result = await desta_tool.process_query(audio_id, question)
            
            # Create result entry
            result_entry = {
                "id": item.get("id", f"question_{i+1}"),
                "audio_id": audio_id,
                "question": question,
                "choices": choices,
                "desta_response": result['final_response'],
                "result_type": result['type'],
                "dataset": item.get("dataset", ""),
                "task": item.get("task", ""),
                "category": item.get("category", ""),
                "sub-category": item.get("sub-category", ""),
                "difficulty": item.get("difficulty", "")
            }
            
            # Add tool information if tools were used
            if result['type'] == 'tool_assisted':
                result_entry["tools_used"] = [call['function'] for call in result['tool_calls']]
                result_entry["initial_response"] = result['initial_response']
            
            results.append(result_entry)
            
            logger.info(f"✅ Completed question {i+1}")
            logger.info(f"Response: {result['final_response'][:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Error processing question {i+1}: {e}")
            # Add error entry
            result_entry = {
                "id": item.get("id", f"question_{i+1}"),
                "audio_id": item.get("audio_id", ""),
                "question": item.get("question", ""),
                "choices": item.get("choices", []),
                "original_answer": item.get("answer", ""),
                "desta_response": f"Error: {str(e)}",
                "result_type": "error",
                "dataset": item.get("dataset", ""),
                "task": item.get("task", ""),
                "category": item.get("category", ""),
                "sub-category": item.get("sub-category", ""),
                "difficulty": item.get("difficulty", "")
            }
            results.append(result_entry)
    
    # Save results to JSON file
    output_file = "mmau-test-mini-desta-results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ Results saved to {output_file}")
        logger.info(f"Total questions processed: {len(results)}")
    except Exception as e:
        logger.error(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    logger.info("Starting DeSTA processing for all questions...")
    asyncio.run(main())