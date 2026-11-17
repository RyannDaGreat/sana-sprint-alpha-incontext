// SANA-Sprint's Gemma prompt template, ported from diffusers
// SanaSprintPipeline._get_gemma_prompt_embeds + encode_prompt (clean_caption=False).

/** SANA's default complex human instruction (diffusers SanaSprintPipeline.__call__). */
export const COMPLEX_HUMAN_INSTRUCTION = [
  "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
  '- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.',
  '- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.',
  'Here are examples of how to transform or refine prompts:',
  '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.',
  '- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.',
  'Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:',
  'User Prompt: ',
];

/**
 * Pure function. Builds the Gemma input ids and attention mask exactly as diffusers does:
 * text = instruction.join("\n") + prompt.lower().strip(), padded right (pad id 0) or truncated
 * to len(encode(instruction)) + maxSequenceLength − 2 tokens, with special tokens (BOS).
 *
 * @param {(text: string) => number[]} encode - Tokenizer encode with special tokens added
 * @param {string} prompt - User prompt
 * @param {string[]} instruction - Complex human instruction lines ([] disables it)
 * @param {number} maxSequenceLength - SANA's 300
 * @param {number} padId - Tokenizer pad id (Gemma: 0)
 * @returns {{ids: number[], mask: number[]}} Both of length totalLength
 *
 * @example
 * const enc = (t) => [2, ...t.split(' ').map((w) => w.length)];
 * gemmaInputs(enc, ' A Cat ', [], 5, 0) // {ids: [2, 1, 3, 0, 0], mask: [1, 1, 1, 0, 0]}
 */
export function gemmaInputs(encode, prompt, instruction, maxSequenceLength, padId) {
  const chi = instruction.join('\n');
  const total = instruction.length ? encode(chi).length + maxSequenceLength - 2 : maxSequenceLength;
  const ids = encode(chi + prompt.toLowerCase().trim()).slice(0, total);
  const mask = ids.map(() => 1);
  while (ids.length < total) { ids.push(padId); mask.push(0); }
  return { ids, mask };
}

/**
 * Pure function. diffusers' select_index = [0] + range(−maxSequenceLength + 1, 0): the BOS token
 * plus the last maxSequenceLength − 1 positions of a length-`total` sequence.
 *
 * @param {number} total - Sequence length fed to Gemma
 * @param {number} maxSequenceLength - SANA's 300
 * @returns {number[]} maxSequenceLength indices into [0, total)
 *
 * @example selectIndex(10, 4) // [0, 7, 8, 9]
 */
export function selectIndex(total, maxSequenceLength) {
  const out = [0];
  for (let i = total - maxSequenceLength + 1; i < total; i++) out.push(i);
  return out;
}
