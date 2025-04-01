import { PuzzleState, PuzzlePiece } from '../types/puzzle';

interface ONNXTensor {
  data: Float32Array | Int32Array | Uint8Array;
  dims: number[];
  type: string;
}

interface ONNXInferenceSession {
  run: (inputs: Record<string, any>) => Promise<Record<string, any>>;
}

export class ImageProcessor {
  private static instance: ImageProcessor | null = null;
  private static isInitializing: boolean = false;
  private static initializationPromise: Promise<void> | null = null;
  private session: ONNXInferenceSession | null = null;
  private isInitialized: boolean = false;
  private ort: any = null;

  private constructor() {
    console.log('ImageProcessor constructor called');
  }

  public static async getInstance(): Promise<ImageProcessor> {
    if (!ImageProcessor.instance) {
      if (!ImageProcessor.isInitializing) {
        ImageProcessor.isInitializing = true;
        ImageProcessor.initializationPromise = new ImageProcessor().initialize();
      }
      await ImageProcessor.initializationPromise;
      ImageProcessor.instance = new ImageProcessor();
      await ImageProcessor.instance.initialize();
    }
    return ImageProcessor.instance;
  }

  public static async cleanup() {
    if (ImageProcessor.instance) {
      ImageProcessor.instance.isInitialized = false;
      ImageProcessor.instance.session = null;
      ImageProcessor.instance = null;
      ImageProcessor.isInitializing = false;
      ImageProcessor.initializationPromise = null;
    }
  }

  private async initialize(): Promise<void> {
    console.log('Starting ImageProcessor initialization...');
    try {
      // Try to load ONNX Runtime Web
      try {
        const ortModule = await import('onnxruntime-web');
        console.log('ONNX Runtime Web module loaded');
        
        // Configure WASM paths
        ortModule.env.wasm.wasmPaths = {
          'ort-wasm-simd-threaded.wasm': '/assets/ort-wasm-simd-threaded.wasm'
        };
        
        console.log('ONNX Runtime Web WASM paths configured');
        
        this.ort = ortModule;
        await this.initializeMobileSAM();
      } catch (error) {
        console.warn('Failed to load ONNX Runtime Web:', error);
        // Continue without ONNX Runtime Web
      }
      
      this.isInitialized = true;
      console.log('ImageProcessor fully initialized');
    } catch (error) {
      console.error('Failed to initialize ImageProcessor:', error);
      throw error;
    }
  }

  private async initializeMobileSAM(): Promise<void> {
    if (!this.ort) {
      console.warn('ONNX Runtime Web not available, skipping MobileSAM initialization');
      return;
    }

    try {
      console.log('Starting MobileSAM initialization...');
      console.log('Attempting to load model from:', './models/mobile_sam.onnx');
      
      // Load the ONNX model
      console.log('Loading ONNX model...');
      const modelResponse = await fetch('./models/mobile_sam.onnx');
      if (!modelResponse.ok) {
        throw new Error('Failed to fetch ONNX model file');
      }
      console.log('ONNX model file fetched successfully');
      
      const modelBuffer = await modelResponse.arrayBuffer();
      console.log('ONNX model buffer size:', modelBuffer.byteLength);
      
      // Convert ArrayBuffer to Uint8Array
      const modelData = new Uint8Array(modelBuffer);
      
      // Try loading with different options
      const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
        executionMode: 'sequential'
      };
      
      console.log('Creating inference session with options:', options);
      
      // Try loading with a simpler configuration first
      try {
        this.session = await this.ort.InferenceSession.create(modelData, {
          executionProviders: ['wasm']
        });
      } catch (error) {
        console.log('Failed with simple config, trying with full config...');
        this.session = await this.ort.InferenceSession.create(modelData, options);
      }
      
      console.log('MobileSAM ONNX model loaded successfully');
    } catch (error: any) {
      console.error('Failed to load MobileSAM:', error);
      throw error;
    }
  }

  private async detectPuzzleGrid(image: HTMLImageElement): Promise<{ corners: any[], size: { rows: number, cols: number } } | null> {
    try {
      console.log('Starting puzzle grid detection with MobileSAM...');
      
      // Preprocess image
      const tensor = await this.preprocessImage(image);
      
      // Generate image embeddings
      const embeddings = await this.generateEmbeddings(tensor);
      
      // Get grid mask using point prompts
      const gridMask = await this.generateMask(embeddings, this.generateGridPoints(image));
      
      // Find grid corners from mask
      const corners = this.findCornersFromMask(gridMask);
      if (!corners) {
        console.log('Failed to find grid corners');
        return null;
      }

      // Get individual tile masks
      const tileMasks = await this.generateMask(embeddings, this.generateTilePoints(corners, image), true);

      // Count tiles to determine grid size
      const { rows, cols } = this.countTiles(tileMasks);

      console.log(`Detected grid size: ${rows}x${cols}`);
      return { corners, size: { rows, cols } };
    } catch (error) {
      console.error('Error detecting puzzle grid:', error);
      return null;
    }
  }

  private async preprocessImage(image: HTMLImageElement): Promise<ONNXTensor> {
    // Convert image to tensor and normalize
    const canvas = document.createElement('canvas');
    canvas.width = 1024; // MobileSAM expects 1024x1024
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');

    // Draw and resize image
    ctx.drawImage(image, 0, 0, 1024, 1024);
    
    // Get image data and convert to tensor
    const imageData = ctx.getImageData(0, 0, 1024, 1024);
    const tensor = new this.ort.Tensor('float32', new Float32Array(imageData.data), [1, 3, 1024, 1024]);
    
    return tensor;
  }

  private async generateEmbeddings(tensor: ONNXTensor): Promise<ONNXTensor> {
    if (!this.session) throw new Error('MobileSAM not initialized');
    
    // Create input in the format expected by the model
    const batched_input = [{
      image: tensor,
      original_size: [1024, 1024],
      point_coords: new this.ort.Tensor('float32', new Float32Array([0.5, 0.5]), [1, 1, 2]),
      point_labels: new this.ort.Tensor('int64', new Int32Array([1]), [1, 1])
    }];
    
    const results = await this.session.run({
      'batched_input': batched_input,
      'multimask_output': new this.ort.Tensor('bool', new Uint8Array([0]), [1])
    });
    
    return results['masks'];
  }

  private async generateMask(embeddings: ONNXTensor, points: number[][], multimask: boolean = false): Promise<any> {
    if (!this.session) throw new Error('MobileSAM not initialized');
    
    // Convert points to tensor format
    const pointsTensor = new this.ort.Tensor('float32', new Float32Array(points.flat()), [1, points.length, 2]);
    
    const results = await this.session.run({
      'batched_input': [{
        image: embeddings,
        original_size: [1024, 1024],
        point_coords: pointsTensor,
        point_labels: new this.ort.Tensor('int64', new Int32Array(points.length).fill(1), [1, points.length])
      }],
      'multimask_output': new this.ort.Tensor('bool', new Uint8Array([multimask ? 1 : 0]), [1])
    });
    
    return results['masks'];
  }

  private generateGridPoints(image: HTMLImageElement): number[][] {
    // Generate points around the edges of the image to detect the grid
    const points = [];
    const step = 50; // Adjust based on image size
    const margin = 20;

    // Add points along the edges
    for (let x = margin; x < image.width - margin; x += step) {
      points.push([x, margin]); // Top edge
      points.push([x, image.height - margin]); // Bottom edge
    }
    for (let y = margin; y < image.height - margin; y += step) {
      points.push([margin, y]); // Left edge
      points.push([image.width - margin, y]); // Right edge
    }

    return points;
  }

  private generateTilePoints(corners: any[], image: HTMLImageElement): number[][] {
    // Generate points inside the grid to detect individual tiles
    const points = [];
    const { width, height } = this.calculateGridDimensions(corners);
    
    // Add points in a grid pattern inside the puzzle area
    for (let y = 0; y < height; y += height / 4) {
      for (let x = 0; x < width; x += width / 4) {
        points.push([x, y]);
      }
    }

    return points;
  }

  private findCornersFromMask(mask: any): number[][] {
    try {
      // Convert mask tensor to canvas
      const canvas = document.createElement('canvas');
      canvas.width = 1024;
      canvas.height = 1024;
      const ctx = canvas.getContext('2d');
      if (!ctx) return [];

      const imageData = new ImageData(
        new Uint8ClampedArray(mask.data.map((v: number) => v > 0.5 ? 255 : 0)),
        1024,
        1024
      );
      ctx.putImageData(imageData, 0, 0);

      // Find contours
      const contours = this.findContours(canvas);
      if (contours.length === 0) return [];

      // Find the largest contour
      let maxArea = 0;
      let maxContour: number[][] = [];
      for (const contour of contours) {
        const area = this.calculateContourArea(contour);
        if (area > maxArea) {
          maxArea = area;
          maxContour = contour;
        }
      }

      // Approximate corners
      return this.approximateCorners(maxContour);
    } catch (error) {
      console.error('Error finding corners from mask:', error);
      return [];
    }
  }

  private findContours(canvas: HTMLCanvasElement): number[][][] {
    const ctx = canvas.getContext('2d');
    if (!ctx) return [];

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const contours: number[][][] = [];
    const visited = new Set<string>();

    // Find starting points (black pixels)
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        if (imageData.data[idx] === 0 && !visited.has(`${x},${y}`)) {
          const contour = this.traceContour(imageData, x, y, canvas.width, canvas.height, visited);
          if (contour.length > 0) {
            contours.push(contour);
          }
        }
      }
    }

    return contours;
  }

  private traceContour(
    imageData: ImageData,
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Set<string>
  ): number[][] {
    const contour: number[][] = [];
    let x = startX;
    let y = startY;
    let dir = 0; // 0: right, 1: down, 2: left, 3: up

    do {
      visited.add(`${x},${y}`);
      contour.push([x, y]);

      // Check 8 neighbors in clockwise order
      const neighbors = [
        [1, 0], [1, 1], [0, 1], [-1, 1],
        [-1, 0], [-1, -1], [0, -1], [1, -1]
      ];

      let found = false;
      for (let i = 0; i < 8; i++) {
        const nextDir = (dir + i) % 8;
        const nx = x + neighbors[nextDir][0];
        const ny = y + neighbors[nextDir][1];

        if (
          nx >= 0 && nx < width &&
          ny >= 0 && ny < height &&
          !visited.has(`${nx},${ny}`)
        ) {
          const idx = (ny * width + nx) * 4;
          if (imageData.data[idx] === 0) {
            x = nx;
            y = ny;
            dir = nextDir;
            found = true;
            break;
          }
        }
      }

      if (!found) break;
    } while (x !== startX || y !== startY);

    return contour;
  }

  private calculateContourArea(contour: number[][]): number {
    let area = 0;
    for (let i = 0; i < contour.length; i++) {
      const j = (i + 1) % contour.length;
      area += contour[i][0] * contour[j][1];
      area -= contour[j][0] * contour[i][1];
    }
    return Math.abs(area) / 2;
  }

  private approximateCorners(contour: number[][]): number[][] {
    // Use Ramer-Douglas-Peucker algorithm to simplify the contour
    const epsilon = 10; // Adjust this value to control simplification
    const simplified = this.ramerDouglasPeucker(contour, epsilon);

    // Find the four corners
    if (simplified.length < 4) return simplified;

    // Sort points by their position (top-left, top-right, bottom-right, bottom-left)
    const center = simplified.reduce(
      (acc, point) => [acc[0] + point[0], acc[1] + point[1]],
      [0, 0]
    ).map(v => v / simplified.length);

    return simplified
      .sort((a, b) => {
        const angleA = Math.atan2(a[1] - center[1], a[0] - center[0]);
        const angleB = Math.atan2(b[1] - center[1], b[0] - center[0]);
        return angleA - angleB;
      })
      .slice(0, 4);
  }

  private ramerDouglasPeucker(points: number[][], epsilon: number): number[][] {
    if (points.length <= 2) return points;

    let maxDist = 0;
    let maxIndex = 0;

    // Find point with maximum distance from line between start and end
    for (let i = 1; i < points.length - 1; i++) {
      const dist = this.pointToLineDistance(points[0], points[points.length - 1], points[i]);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }

    if (maxDist > epsilon) {
      // Recursively simplify the two sub-paths
      const firstHalf = this.ramerDouglasPeucker(points.slice(0, maxIndex + 1), epsilon);
      const secondHalf = this.ramerDouglasPeucker(points.slice(maxIndex), epsilon);
      return [...firstHalf.slice(0, -1), ...secondHalf];
    }

    return [points[0], points[points.length - 1]];
  }

  private pointToLineDistance(lineStart: number[], lineEnd: number[], point: number[]): number {
    const [x1, y1] = lineStart;
    const [x2, y2] = lineEnd;
    const [x0, y0] = point;

    const numerator = Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
    const denominator = Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));

    return numerator / denominator;
  }

  private sortCorners(corners: any[]): any[] {
    // Find the center point
    const center = corners.reduce((acc, corner) => ({
      x: acc.x + corner.x / 4,
      y: acc.y + corner.y / 4
    }), { x: 0, y: 0 });

    // Sort corners by angle relative to center
    return corners.sort((a, b) => {
      const angleA = Math.atan2(a.y - center.y, a.x - center.x);
      const angleB = Math.atan2(b.y - center.y, b.x - center.x);
      return angleA - angleB;
    });
  }

  private countTiles(masks: any[]): { rows: number, cols: number } {
    try {
      // Convert masks to binary images
      const binaryMasks = masks.map(mask => {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        const imageData = new ImageData(
          new Uint8ClampedArray(mask.data.map((v: number) => v > 0.5 ? 255 : 0)),
          1024,
          1024
        );
        ctx.putImageData(imageData, 0, 0);
        return canvas;
      }).filter((canvas): canvas is HTMLCanvasElement => canvas !== null);

      // Find connected components in the masks
      const components = this.findConnectedComponents(binaryMasks);
      
      // Count rows and columns based on component positions
      const { rows, cols } = this.countGridDimensions(components);

      return { rows, cols };
    } catch (error) {
      console.error('Error counting tiles:', error);
      return { rows: 4, cols: 4 }; // Default to 4x4
    }
  }

  private findConnectedComponents(masks: HTMLCanvasElement[]): any[] {
    const components: any[] = [];
    const visited = new Set<string>();

    for (let i = 0; i < masks.length; i++) {
      const mask = masks[i];
      const ctx = mask.getContext('2d');
      if (!ctx) continue;

      const imageData = ctx.getImageData(0, 0, mask.width, mask.height);
      
      for (let y = 0; y < mask.height; y++) {
        for (let x = 0; x < mask.width; x++) {
          const key = `${i},${x},${y}`;
          if (!visited.has(key)) {
            const idx = (y * mask.width + x) * 4;
            if (imageData.data[idx] > 0) {
              const component = this.floodFill(imageData, x, y, mask.width, mask.height, visited, i);
              if (component.points.length > 100) { // Filter out noise
                components.push(component);
              }
            }
          }
        }
      }
    }

    return components;
  }

  private floodFill(
    imageData: ImageData,
    startX: number,
    startY: number,
    width: number,
    height: number,
    visited: Set<string>,
    maskIndex: number
  ): { maskIndex: number, points: number[][], bounds: { minX: number, minY: number, maxX: number, maxY: number } } {
    const points: number[][] = [];
    const stack: number[][] = [[startX, startY]];
    const bounds = {
      minX: startX,
      minY: startY,
      maxX: startX,
      maxY: startY
    };

    while (stack.length > 0) {
      const [x, y] = stack.pop()!;
      const key = `${maskIndex},${x},${y}`;
      
      if (visited.has(key)) continue;
      visited.add(key);

      const idx = (y * width + x) * 4;
      if (imageData.data[idx] === 0) continue;

      points.push([x, y]);
      bounds.minX = Math.min(bounds.minX, x);
      bounds.minY = Math.min(bounds.minY, y);
      bounds.maxX = Math.max(bounds.maxX, x);
      bounds.maxY = Math.max(bounds.maxY, y);

      // Add neighbors to stack
      if (x > 0) stack.push([x - 1, y]);
      if (x < width - 1) stack.push([x + 1, y]);
      if (y > 0) stack.push([x, y - 1]);
      if (y < height - 1) stack.push([x, y + 1]);
    }

    return { maskIndex, points, bounds };
  }

  private countGridDimensions(components: any[]): { rows: number, cols: number } {
    // Sort components by position
    const sortedByY = [...components].sort((a, b) => a.bounds.minY - b.bounds.minY);
    const sortedByX = [...components].sort((a, b) => a.bounds.minX - b.bounds.minX);

    // Count unique rows and columns
    const rows = new Set<number>();
    const cols = new Set<number>();

    for (const component of components) {
      const centerY = (component.bounds.minY + component.bounds.maxY) / 2;
      const centerX = (component.bounds.minX + component.bounds.maxX) / 2;

      // Find closest row and column
      let minRowDist = Infinity;
      let minColDist = Infinity;
      let rowIndex = 0;
      let colIndex = 0;

      sortedByY.forEach((row, i) => {
        const rowCenter = (row.bounds.minY + row.bounds.maxY) / 2;
        const dist = Math.abs(centerY - rowCenter);
        if (dist < minRowDist) {
          minRowDist = dist;
          rowIndex = i;
        }
      });

      sortedByX.forEach((col, i) => {
        const colCenter = (col.bounds.minX + col.bounds.maxX) / 2;
        const dist = Math.abs(centerX - colCenter);
        if (dist < minColDist) {
          minColDist = dist;
          colIndex = i;
        }
      });

      rows.add(rowIndex);
      cols.add(colIndex);
    }

    return {
      rows: rows.size,
      cols: cols.size
    };
  }

  private extractTileFromMask(image: HTMLImageElement, mask: any): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

    // Create a temporary canvas for the mask
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = 1024;
    maskCanvas.height = 1024;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return canvas;

    // Draw the mask
    const maskImageData = new ImageData(
      new Uint8ClampedArray(mask.data.map((v: number) => v > 0.5 ? 255 : 0)),
      1024,
      1024
    );
    maskCtx.putImageData(maskImageData, 0, 0);

    // Draw the original image
    ctx.drawImage(image, 0, 0, 1024, 1024);

    // Apply the mask
    const imageData = ctx.getImageData(0, 0, 1024, 1024);
    const maskData = maskCtx.getImageData(0, 0, 1024, 1024);

    for (let i = 0; i < imageData.data.length; i += 4) {
      if (maskData.data[i] === 0) {
        imageData.data[i] = 0;     // R
        imageData.data[i + 1] = 0; // G
        imageData.data[i + 2] = 0; // B
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  private calculateGridDimensions(corners: any[]): { width: number, height: number } {
    // Calculate width and height of the grid from corners
    return {
      width: Math.max(
        Math.abs(corners[1].x - corners[0].x),
        Math.abs(corners[2].x - corners[3].x)
      ),
      height: Math.max(
        Math.abs(corners[3].y - corners[0].y),
        Math.abs(corners[2].y - corners[1].y)
      )
    };
  }

  private async extractTiles(image: HTMLImageElement, grid: { corners: any[], size: { rows: number, cols: number } }): Promise<number[][]> {
    try {
      const { corners, size } = grid;
      const { rows, cols } = size;

      // Preprocess image
      const tensor = await this.preprocessImage(image);
      
      // Generate embeddings
      const embeddings = await this.generateEmbeddings(tensor);
      
      // Get individual tile masks
      const tileMasks = await this.generateMask(embeddings, this.generateTilePoints(corners, image), true);

      // Process each tile mask to extract the tile image and compute its hash
      const tiles: number[][] = [];
      for (let row = 0; row < rows; row++) {
        const rowTiles: number[] = [];
        for (let col = 0; col < cols; col++) {
          const tileMask = tileMasks[row * cols + col];
          const tileImage = this.extractTileFromMask(image, tileMask);
          const hash = this.hashTile(tileImage);
          rowTiles.push(hash);
        }
        tiles.push(rowTiles);
      }

      return tiles;
    } catch (error) {
      console.error('Error extracting tiles:', error);
      return [];
    }
  }

  private hashTile(tile: HTMLCanvasElement): number {
    // Improved hash function for tile images
    const ctx = tile.getContext('2d');
    if (!ctx) return 0;

    const imageData = ctx.getImageData(0, 0, tile.width, tile.height);
    let hash = 0;
    
    // Use a more sophisticated hashing algorithm
    for (let i = 0; i < imageData.data.length; i += 4) {
      hash = ((hash << 5) - hash) + imageData.data[i];
      hash = hash & hash;
    }
    
    return Math.abs(hash) % 16; // For 4x4 grid
  }

  public async processImage(image: HTMLImageElement): Promise<{ state: PuzzleState }> {
    console.log('Starting image processing...');
    if (!this.isInitialized) {
      console.error('ImageProcessor not initialized');
      throw new Error('ImageProcessor not initialized');
    }

    try {
      console.log('Image dimensions:', image.width, 'x', image.height);

      // Detect puzzle grid
      console.log('Detecting puzzle grid...');
      const grid = await this.detectPuzzleGrid(image);
      if (!grid) {
        throw new Error('Failed to detect puzzle grid');
      }
      console.log('Grid detected:', grid);

      // Extract tiles
      console.log('Extracting tiles...');
      const tiles = await this.extractTiles(image, grid);
      if (tiles.length === 0) {
        throw new Error('Failed to extract tiles');
      }
      console.log('Tiles extracted:', tiles);

      // Find empty position
      let emptyRow = 0, emptyCol = 0;
      for (let row = 0; row < tiles.length; row++) {
        for (let col = 0; col < tiles[row].length; col++) {
          if (tiles[row][col] === 0) {
            emptyRow = row;
            emptyCol = col;
            break;
          }
        }
      }

      // Create puzzle state
      const state: PuzzleState = {
        grid: tiles,
        emptyPosition: { row: emptyRow, col: emptyCol },
        size: grid.size
      };

      console.log('Returning state:', state);
      return { state };
    } catch (error) {
      console.error('Error processing image:', error);
      throw error;
    }
  }

  public async processFrame(video: HTMLVideoElement): Promise<PuzzleState | null> {
    console.log('Processing frame...');
    if (!this.isInitialized) {
      console.error('ImageProcessor not initialized');
      return null;
    }

    try {
      // Create a canvas to capture the current frame
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log('Canvas created with dimensions:', canvas.width, 'x', canvas.height);
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.error('Failed to get canvas context');
        return null;
      }

      // Draw the current frame to the canvas
      ctx.drawImage(video, 0, 0);
      console.log('Frame drawn to canvas');

      // Convert canvas to image and process
      const image = new Image();
      image.src = canvas.toDataURL('image/png');
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
      });

      const result = await this.processImage(image);
      return result.state;
    } catch (error) {
      console.error('Error processing frame:', error);
      return null;
    }
  }

  public async detectPuzzleInAR(video: HTMLVideoElement): Promise<{ state: PuzzleState } | null> {
    console.log('Starting AR puzzle detection...');
    if (!this.isInitialized) {
      console.error('ImageProcessor not initialized');
      return null;
    }

    try {
      // This will be implemented with AR detection logic
      return null;
    } catch (error) {
      console.error('Error in AR detection:', error);
      return null;
    }
  }
} 