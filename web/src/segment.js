// Photo → matte: foreground segmentation with BEN2 (MIT, onnx-community/BEN2-ONNX, fp16, 219 MB) on
// ONNX Runtime Web. The model file comes from the Hugging Face hub through the app's OPFS cache.
// Why BEN2 and not BiRefNet (MIT too, slightly cleaner masks on CPU): BiRefNet's ONNX has Split nodes
// with up to 33 outputs, more storage buffers than WebGPU allows per shader (failed on the test
// adapter, max 10), and its fp32 run ran out of WASM memory (std::bad_alloc). BEN2 has neither problem.
// The model sees the photo stretched to 1024²; its soft alpha is resized back to the photo's aspect
// and letterboxed into the matte the same way as an uploaded matte (fit inside, centered).

import * as ort from 'onnxruntime-web/webgpu';
import { fetchCached } from './models.js';
import { makeTensor, asFloat32 } from './pipeline.js';
import { SIZE, fitRect } from './util.js';

// fp16 weights with float32 input/output; on WebGPU it needs shader-f16, else main.js picks WASM.
export const SEG_FILE = {
  name: 'BEN2 model_fp16.onnx', bytes: 219121675,
  url: 'https://huggingface.co/onnx-community/BEN2-ONNX/resolve/main/onnx/model_fp16.onnx',
  key: 'https://huggingface.co/onnx-community/BEN2-ONNX/resolve/main/onnx/model_fp16.onnx?v=dfdc25f421f32a0d1268e0f2ff2153d340e8f1d52d3dd16f5dc33c1ce85cedf1',
};
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/**
 * Pure function. RGBA pixels → planar ImageNet-normalized RGB tensor.
 *
 * @param {Uint8ClampedArray} rgba - (H, W, 4) uint8
 * @param {number} hw - H · W
 * @returns {Float32Array} (1, 3, H, W) float32, (v / 255 − mean) / std per channel
 *
 * @example photoToTensor(new Uint8ClampedArray([255, 0, 124, 255]), 1)
 * // Float32Array [2.249, -2.036, 0.357]  (rounded)
 */
export function photoToTensor(rgba, hw) {
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    for (let c = 0; c < 3; c++) out[c * hw + i] = (rgba[4 * i + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
  }
  return out;
}

/**
 * Pure function. Alpha in [0, 1] → opaque gray RGBA (white = foreground).
 *
 * @param {Float32Array} alpha - (1, 1, H, W) float32
 * @returns {Uint8ClampedArray} (H, W, 4) uint8, R = G = B = round(alpha · 255)
 *
 * @example alphaToGray(new Float32Array([0.5, 1, 0])) // [128,128,128,255, 255,255,255,255, 0,0,0,255]
 */
export function alphaToGray(alpha) {
  const out = new Uint8ClampedArray(4 * alpha.length);
  for (let i = 0; i < alpha.length; i++) {
    out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = alpha[i] * 255;
    out[4 * i + 3] = 255;
  }
  return out;
}

export class Segmenter {
  /**
   * Command. Downloads (or reads from the OPFS cache) the BEN2 file and creates the session.
   *
   * @param {'webgpu'|'wasm'} ep
   * @param {(file: object, done: number, total: number, fromCache: boolean) => void} onProgress
   * @returns {Promise<Segmenter>}
   */
  static async load(ep, onProgress) {
    const s = new Segmenter();
    const data = await fetchCached(SEG_FILE, (d, t, c) => onProgress(SEG_FILE, d, t, c));
    s.session = await ort.InferenceSession.create(data, { executionProviders: [ep] });
    s.input = s.session.inputMetadata[0];
    return s;
  }

  /**
   * Command (runs the model). Photo → soft matte at the photo's fitted size.
   *
   * @param {ImageBitmap} bitmap - Any size
   * @returns {Promise<{gray: Uint8ClampedArray, w: number, h: number, ms: number}>} gray (h, w, 4) uint8
   */
  async run(bitmap) {
    const sq = new OffscreenCanvas(SIZE, SIZE).getContext('2d', { willReadFrequently: true });
    sq.drawImage(bitmap, 0, 0, SIZE, SIZE); // stretched, as the model was trained
    const t0 = performance.now();
    const out = await this.session.run({
      [this.input.name]: makeTensor(this.input.type, photoToTensor(sq.getImageData(0, 0, SIZE, SIZE).data, SIZE * SIZE), [1, 3, SIZE, SIZE]),
    });
    const ms = performance.now() - t0;
    sq.putImageData(new ImageData(alphaToGray(asFloat32(out[this.session.outputNames[0]])), SIZE, SIZE), 0, 0);
    const { w, h } = fitRect(bitmap.width, bitmap.height);
    const fit = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true });
    fit.drawImage(sq.canvas, 0, 0, w, h);
    return { gray: fit.getImageData(0, 0, w, h).data, w, h, ms };
  }
}
