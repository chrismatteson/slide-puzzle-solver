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
  private encoderSession: ONNXInferenceSession | null = null;
  private isInitialized: boolean = false;
  private ort: any = null;
  private image: HTMLImageElement | null = null;

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
      // Try to use the globally loaded ONNX Runtime Web
      try {
        console.log('Checking for globally loaded ONNX Runtime Web...');
        
        // @ts-ignore
        if (window.ort) {
          console.log('Found globally loaded ONNX Runtime Web');
          // @ts-ignore
          this.ort = window.ort;
          await this.initializeMobileSAM();
        } else {
          console.warn('ONNX Runtime Web not found in global scope');
        }
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
      
      // MobileSAM requires two models: an encoder and a decoder
      // The encoder processes the image to image_embeddings
      // The decoder takes those embeddings plus points to create a mask
      
      // 1. Load the encoder model
      console.log('Attempting to load encoder model from: ./models/mobilesam.encoder.onnx');
      const encoderResponse = await fetch('./models/mobilesam.encoder.onnx');
      if (!encoderResponse.ok) {
        throw new Error(`Failed to fetch encoder model: ${encoderResponse.status} ${encoderResponse.statusText}`);
      }
      
      console.log('Encoder model fetched successfully');
      const encoderBuffer = await encoderResponse.arrayBuffer();
      console.log('Encoder model buffer size:', (encoderBuffer.byteLength / (1024 * 1024)).toFixed(2) + ' MB');
      
      // Create encoder session
      this.encoderSession = await this.ort.InferenceSession.create(new Uint8Array(encoderBuffer), {
        executionProviders: ['wasm']
      });
      console.log('Encoder session created successfully');
      
      // 2. Load the decoder model
      console.log('Attempting to load decoder model from: ./models/mobile_sam.onnx');
      
      const decoderResponse = await fetch('./models/mobile_sam.onnx');
      if (!decoderResponse.ok) {
        throw new Error(`Failed to fetch decoder model: ${decoderResponse.status} ${decoderResponse.statusText}`);
      }
      console.log('Decoder model fetched successfully');
      
      const decoderBuffer = await decoderResponse.arrayBuffer();
      console.log('Decoder model buffer size:', (decoderBuffer.byteLength / (1024 * 1024)).toFixed(2) + ' MB');
      
      // Create decoder session
      this.session = await this.ort.InferenceSession.create(new Uint8Array(decoderBuffer), {
        executionProviders: ['wasm']
      });
      console.log('Decoder model loaded successfully');
      
      // Log model metadata without running test inference
      await this.safeLogModelDetails();
      
      console.log('MobileSAM initialized successfully');
    } catch (error: any) {
      console.error('Failed to load MobileSAM:', error);
      throw error;
    }
  }

  private validateGridSize(corners: number[][]): { isValid: boolean, gridWidth: number, gridHeight: number } {
    // Calculate grid dimensions
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    for (const corner of corners) {
      minX = Math.min(minX, corner[0]);
      minY = Math.min(minY, corner[1]);
      maxX = Math.max(maxX, corner[0]);
      maxY = Math.max(maxY, corner[1]);
    }
    
    const gridWidth = maxX - minX;
    const gridHeight = maxY - minY;
    
    console.log(`Grid dimensions check: ${gridWidth}x${gridHeight} pixels`);
    
    // If the grid is too small, it's probably not valid
    // But if we're getting a default grid (from our fallback), it should be valid
    const isDefaultGrid = 
      (corners.length === 4 && 
       corners[0][0] === 100 && corners[0][1] === 100 &&
       corners[1][0] === 924 && corners[1][1] === 100);
    
    if (isDefaultGrid) {
      console.log('Using default grid - bypassing size validation');
      return { isValid: true, gridWidth, gridHeight };
    }
    
    // A reasonable puzzle grid should be at least 30x30 pixels (more lenient than before)
    // This prevents tiny false positives
    const isValid = gridWidth >= 30 && gridHeight >= 30 && 
                    gridWidth <= 1000 && gridHeight <= 1000 &&
                    // Also check that the grid is somewhat square-like, but be more lenient
                    gridWidth / gridHeight < 3 && gridHeight / gridWidth < 3;
    
    return { isValid, gridWidth, gridHeight };
  }
  
  private async detectPuzzleGrid(image: HTMLImageElement): Promise<{ corners: any[], size: { rows: number, cols: number } } | null> {
    try {
      console.log('Starting puzzle grid detection...');
      
      // Try our specialized slide puzzle detector first
      console.log('Using slide puzzle square detector...');
      const gridDetection = await this.detectSlidePuzzleGrid(image);
      if (gridDetection) {
        console.log('Slide puzzle grid detected successfully!');
        return gridDetection;
      }
      
      // If that fails, try using MobileSAM
      if (this.session) {
        console.log('Slide puzzle detector failed, trying MobileSAM for detection');
        // MobileSAM approach
        try {
          console.log('Step 1: Preprocess image');
          // Preprocess image
          const tensor = await this.preprocessImage(image);
          
          try {
            console.log('Step 2: Generate image embeddings (uses dummy in this implementation)');
            // Generate image embeddings
            const embeddings = await this.generateEmbeddings(tensor);
            
            try {
              console.log('Step 3: Generate grid mask using point prompts');
              // Get grid mask using point prompts
              const gridPoints = this.generateGridPoints(image);
              console.log('Grid points for prompting:', gridPoints);
              const gridMask = await this.generateMask(embeddings, gridPoints);
              
              try {
                console.log('Step 4: Find grid corners from mask');
                // Find grid corners from mask
                const corners = this.findCornersFromMask(gridMask);
                if (!corners || corners.length < 4) {
                  console.log('Failed to find grid corners using MobileSAM, falling back to basic detection');
                  throw new Error('Failed to find grid corners');
                }

                console.log('Found grid corners:', corners);

                // Validate that the grid is a reasonable size
                const { isValid, gridWidth, gridHeight } = this.validateGridSize(corners);
                if (!isValid) {
                  console.log('Detected grid is too small or invalid, falling back to basic detection');
                  // Instead of rejecting, try basic detection as a fallback
                  return this.detectPuzzleGridBasic(image);
                }
                
                console.log(`Valid grid detected with dimensions: ${gridWidth}x${gridHeight} pixels`);

                try {
                  console.log('Step 5: Generate tile masks to determine grid size');
                  // Get individual tile masks
                  const tilePoints = this.generateTilePoints(corners, image);
                  console.log('Tile points for prompting:', tilePoints);
                  const tileMasks = await this.generateMask(embeddings, tilePoints, true);

                  // Count tiles to determine grid size
                  const { rows, cols } = this.countTiles(tileMasks);

                  console.log(`Detected grid size: ${rows}x${cols}`);
                  return { corners, size: { rows, cols } };
                } catch (error) {
                  console.warn('Error counting tiles:', error);
                  // Fall back to default grid size
                  console.log('Using fallback grid size: 4x4');
                  return { corners, size: { rows: 4, cols: 4 } };
                }
              } catch (error) {
                console.warn('Error finding corners from mask:', error);
                throw error; // Let it fall through to basic detection
              }
            } catch (error) {
              console.warn('Error generating mask:', error);
              throw error; // Let it fall through to basic detection
            }
          } catch (error) {
            console.warn('Error generating embeddings:', error);
            throw error; // Let it fall through to basic detection
          }
        } catch (error) {
          console.warn('MobileSAM detection failed:', error);
          // Fall through to basic detection
        }
      } else {
        console.log('MobileSAM not initialized, using basic detection');
      }
      
      // Fallback to basic approach
      console.log('Using basic detection approach');
      return this.detectPuzzleGridBasic(image);
    } catch (error) {
      console.error('Error detecting puzzle grid:', error);
      // Try basic detection as a final fallback
      try {
        return this.detectPuzzleGridBasic(image);
      } catch (e) {
        // If even that fails, return null
        console.log('No valid puzzle grid detected, even with basic detection');
        return null;
      }
    }
  }
  
  private detectPuzzleGridBasic(image: HTMLImageElement): { corners: any[], size: { rows: number, cols: number } } | null {
    console.log('Performing basic puzzle grid detection');
    
    // Look for a square region in the center part of the image
    // rather than using the entire image
    
    const width = image.width;
    const height = image.height;
    
    // Calculate a centered square region that's 70% of the smaller dimension
    const squareSize = Math.min(width, height) * 0.7;
    const centerX = width / 2;
    const centerY = height / 2;
    const halfSize = squareSize / 2;
    
    const corners = [
      [centerX - halfSize, centerY - halfSize], // top-left
      [centerX + halfSize, centerY - halfSize], // top-right
      [centerX + halfSize, centerY + halfSize], // bottom-right
      [centerX - halfSize, centerY + halfSize]  // bottom-left
    ];
    
    console.log(`Using centered square region: center=(${centerX},${centerY}), size=${squareSize}`);
    
    // Validate the grid size - this is mostly a formality since we calculated a reasonable size
    const { isValid } = this.validateGridSize(corners);
    if (!isValid) {
      console.log('Basic detection failed: calculated grid dimensions are invalid');
      return null;
    }
    
    // Default to 4x4 grid for sliding puzzle
    const size = { rows: 4, cols: 4 };
    
    console.log('Basic detection completed with reasonable default values');
    return { corners, size };
  }

  private async preprocessImage(image: HTMLImageElement): Promise<ONNXTensor> {
    try {
      // Create a canvas to resize the image to 1024x1024
      const canvas = document.createElement('canvas');
      canvas.width = 1024;
      canvas.height = 1024;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }
      
      console.log(`Original image dimensions: ${image.width}x${image.height}`);
      
      // Draw the original image onto the canvas, preserving aspect ratio
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
      const scaledWidth = image.width * scale;
      const scaledHeight = image.height * scale;
      const offsetX = (canvas.width - scaledWidth) / 2;
      const offsetY = (canvas.height - scaledHeight) / 2;
      
      ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);
      
      // Get the pixel data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Create the tensor data with shape [3, 3, 1024]
      const float32Data = new Float32Array(3 * 3 * 1024);
      
      // Fill with simple data pattern - 3 channels, 3 values per channel, 1024 items
      for (let x = 0; x < 1024; x++) {
        // Get a pixel value from the middle of the image
        const pixelIndex = (512 * canvas.width + x) * 4;
        
        // Set the same value for all 3 positions in each channel 
        float32Data[0 * 3 * 1024 + 0 * 1024 + x] = imageData.data[pixelIndex] / 255.0;
        float32Data[0 * 3 * 1024 + 1 * 1024 + x] = imageData.data[pixelIndex] / 255.0;
        float32Data[0 * 3 * 1024 + 2 * 1024 + x] = imageData.data[pixelIndex] / 255.0;
        
        float32Data[1 * 3 * 1024 + 0 * 1024 + x] = imageData.data[pixelIndex + 1] / 255.0;
        float32Data[1 * 3 * 1024 + 1 * 1024 + x] = imageData.data[pixelIndex + 1] / 255.0;
        float32Data[1 * 3 * 1024 + 2 * 1024 + x] = imageData.data[pixelIndex + 1] / 255.0;
        
        float32Data[2 * 3 * 1024 + 0 * 1024 + x] = imageData.data[pixelIndex + 2] / 255.0;
        float32Data[2 * 3 * 1024 + 1 * 1024 + x] = imageData.data[pixelIndex + 2] / 255.0;
        float32Data[2 * 3 * 1024 + 2 * 1024 + x] = imageData.data[pixelIndex + 2] / 255.0;
      }
      
      console.log('Creating tensor with shape [3, 3, 1024]');
      return new this.ort.Tensor('float32', float32Data, [3, 3, 1024]);
    } catch (error) {
      console.error('Error preprocessing image:', error);
      throw error;
    }
  }

  private async generateEmbeddings(tensor: ONNXTensor): Promise<ONNXTensor> {
    // Make sure we have an encoder session
    if (!this.encoderSession) {
      throw new Error('Encoder model not available');
    }
    
    console.log('Using encoder model to generate embeddings');
    
    // Try to get input name information from encoder session
    let inputName = 'input_image';
    try {
      // @ts-ignore - Access internal properties if available
      if (this.encoderSession.inputNames && this.encoderSession.inputNames.length > 0) {
        // @ts-ignore
        inputName = this.encoderSession.inputNames[0];
        console.log('Using detected encoder input name:', inputName);
      }
    } catch (error) {
      console.log('Could not access encoder input names, using default:', inputName);
    }
    
    // Create feeds for the encoder
    const feeds = {
      [inputName]: tensor
    };
    
    console.log('Running encoder inference with input shape:', tensor.dims);
    const results = await this.encoderSession.run(feeds);
    console.log('Encoder inference completed. Available outputs:', Object.keys(results));
    
    // Select the output - usually there's just one for an encoder
    const outputKeys = Object.keys(results);
    if (outputKeys.length > 0) {
      const outputTensor = results[outputKeys[0]];
      console.log('Using encoder output:', outputKeys[0], 'with shape:', outputTensor.dims);
      return outputTensor;
    }
    
    throw new Error('No output tensors found from encoder');
  }

  private async generateMask(embeddings: ONNXTensor, points: number[][], multimask: boolean = false): Promise<HTMLCanvasElement[]> {
    console.log(`Generating mask from points: ${JSON.stringify(points)}`);
    
    try {
      if (!this.session) {
        throw new Error('Decoder session not initialized');
      }
      
      // Convert points to tensor format
      const pointsArray = new Float32Array(points.length * 2);
      points.forEach((point, i) => {
        pointsArray[i * 2] = point[0];
        pointsArray[i * 2 + 1] = point[1];
      });
      
      // All points are foreground (labeled as 1)
      const labelsArray = new Float32Array(points.length);
      for (let i = 0; i < points.length; i++) {
        labelsArray[i] = 1; // 1 means foreground
      }
      
      // Create tensors
      const pointCoordsTensor = new this.ort.Tensor('float32', pointsArray, [1, points.length, 2]);
      const pointLabelsTensor = new this.ort.Tensor('float32', labelsArray, [1, points.length]);
      
      // Empty mask input (no previous mask)
      const maskInput = new this.ort.Tensor(
        'float32', 
        new Float32Array(1 * 1 * 256 * 256).fill(0),
        [1, 1, 256, 256]
      );
      
      // Has mask input flag - set to 0 (no previous mask)
      const hasMaskInput = new this.ort.Tensor('float32', new Float32Array([0]), [1]);
      
      // Original image size
      const origSizeArray = new Float32Array([1024, 1024]);
      const origSizeTensor = new this.ort.Tensor('float32', origSizeArray, [2]);
      
      // Create inputs for the decoder model
      const feeds = {
        'image_embeddings': embeddings,
        'point_coords': pointCoordsTensor,
        'point_labels': pointLabelsTensor,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
        'orig_im_size': origSizeTensor,
        'multimask_output': new this.ort.Tensor('float32', new Float32Array([multimask ? 1.0 : 0.0]), [1])
      };
      
      console.log(`Running inference to generate mask with inputs: ${Object.keys(feeds)}`);
      
      // Run inference
      const results = await this.session.run(feeds);
      
      console.log(`Mask generation completed. Available outputs: ${Object.keys(results)}`);
      
      // Check if we got masks in the output
      if (!results.masks) {
        console.warn('No masks found in model output');
        // Check what outputs we actually got
        const availableOutputs = Object.keys(results);
        if (availableOutputs.length > 0) {
          console.log('Available outputs:', availableOutputs);
          // Try alternative output names
          const possibleMaskOutputs = ['masks', 'mask', 'segmentation', 'segmentations', 'output_mask'];
          for (const outputName of possibleMaskOutputs) {
            if (results[outputName]) {
              console.log(`Found alternative mask output: ${outputName}`);
              results.masks = results[outputName];
              break;
            }
          }
        }
        
        // If we still don't have masks, check if we have low_res_masks
        if (results.low_res_masks && !results.masks) {
          console.log('Using low_res_masks as fallback');
          results.masks = results.low_res_masks;
        }
        
        // If we still don't have masks, we can't proceed
        if (!results.masks) {
          console.error('No mask output found in model results');
          return [];
        }
      }
      
      // Convert the mask tensor to a canvas
      // First, get the mask data and dimensions
      const masksTensor = results.masks;
      const maskData = masksTensor.data;
      const maskDims = masksTensor.dims;
      
      console.log(`Mask tensor shape: [${maskDims}]`);
      console.log(`Mask data length: ${maskData.length}`);
      
      // Check if we got multiple masks (batch size > 1)
      const numMasks = multimask ? maskDims[1] : 1;
      const maskHeight = maskDims[2];
      const maskWidth = maskDims[3];
      
      console.log(`Converting ${numMasks} masks to canvas, dimensions: ${maskWidth}x${maskHeight}`);
      
      // Create canvases for each mask
      const maskCanvases: HTMLCanvasElement[] = [];
      
      for (let m = 0; m < numMasks; m++) {
        const canvas = document.createElement('canvas');
        canvas.width = maskWidth;
        canvas.height = maskHeight;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) continue;
        
        // Create an ImageData object to hold the mask
        const imageData = ctx.createImageData(maskWidth, maskHeight);
        const pixelData = imageData.data;
        
        // Fill the image data with the mask values
        for (let y = 0; y < maskHeight; y++) {
          for (let x = 0; x < maskWidth; x++) {
            // Calculate the index in the mask tensor data
            // Format is [batch, mask_id, height, width]
            const tensorIdx = (0 * numMasks * maskHeight * maskWidth) + 
                             (m * maskHeight * maskWidth) + 
                             (y * maskWidth) + 
                             x;
            
            // Get the mask value (0-1 float)
            const maskValue = maskData[tensorIdx];
            
            // Convert to 0-255 range and use as alpha
            const value = Math.round(maskValue * 255);
            
            // Calculate the index in the pixel data
            const pixelIdx = (y * maskWidth + x) * 4;
            
            // Set white with the calculated alpha
            pixelData[pixelIdx] = 255;     // R
            pixelData[pixelIdx + 1] = 255; // G
            pixelData[pixelIdx + 2] = 255; // B
            pixelData[pixelIdx + 3] = value; // A
          }
        }
        
        // Put the image data onto the canvas
        ctx.putImageData(imageData, 0, 0);
        
        // Resize the mask to 1024x1024 to match the input image size
        const resizedCanvas = document.createElement('canvas');
        resizedCanvas.width = 1024;
        resizedCanvas.height = 1024;
        
        const resizedCtx = resizedCanvas.getContext('2d');
        if (!resizedCtx) continue;
        
        // Draw the mask onto the resized canvas
        resizedCtx.drawImage(canvas, 0, 0, maskWidth, maskHeight, 0, 0, 1024, 1024);
        
        maskCanvases.push(resizedCanvas);
      }
      
      console.log(`Generated ${maskCanvases.length} mask canvases`);
      
      return maskCanvases;
    } catch (error) {
      console.error('Error generating mask:', error);
      return [];
    }
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

  private generateTilePoints(corners: number[][], image: HTMLImageElement): number[][] {
    // Generate points inside the grid to detect individual tiles
    const points = [];
    
    // Make sure we have 4 corners
    if (corners.length < 4) {
      console.log('Not enough corners detected for tile points generation');
      return [];
    }
    
    try {
      // Calculate the grid bounds based on the corners
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      
      for (const corner of corners) {
        minX = Math.min(minX, corner[0]);
        minY = Math.min(minY, corner[1]);
        maxX = Math.max(maxX, corner[0]);
        maxY = Math.max(maxY, corner[1]);
      }
      
      // Calculate width and height
      const width = maxX - minX;
      const height = maxY - minY;
      
      if (width <= 0 || height <= 0) {
        console.log('Invalid grid dimensions:', width, height);
        return [];
      }
      
      console.log('Grid dimensions:', width, 'x', height);
      
      // Add points in a grid pattern inside the puzzle area
      // Generate a 5x5 grid of points (for a 4x4 puzzle)
      const rows = 5, cols = 5;
      
      for (let r = 1; r < rows; r++) {
        for (let c = 1; c < cols; c++) {
          const x = minX + (width * c) / cols;
          const y = minY + (height * r) / rows;
          points.push([x, y]);
        }
      }
      
      console.log(`Generated ${points.length} tile points`);
    } catch (error) {
      console.error('Error generating tile points:', error);
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

      // Create a visualization of the mask for debugging
      console.log('Creating visualization of mask');
      const maskedArea = mask.data.filter((v: number) => v > 0.5).length;
      const totalArea = mask.data.length;
      const percentCovered = (maskedArea / totalArea) * 100;
      console.log(`Mask covers ${percentCovered.toFixed(2)}% of the image (${maskedArea} / ${totalArea} pixels)`);

      // If too much of the image is masked, it's likely the whole image or a bad detection
      if (percentCovered > 90) {
        console.log('Mask covers too much of the image, likely a bad detection');
        // Use simpler approach: assume grid is centered in the image with padding
        return [
          [100, 100],
          [924, 100],
          [924, 924],
          [100, 924]
        ];
      } else if (percentCovered < 35) {
        console.log('Mask covers too little of the image, likely missed the puzzle');
      }

      const imageData = new ImageData(
        new Uint8ClampedArray(mask.data.map((v: number) => v > 0.5 ? 255 : 0)),
        1024,
        1024
      );
      ctx.putImageData(imageData, 0, 0);

      // Find contours
      const contours = this.findContours(canvas);
      console.log(`Found ${contours.length} contours in the mask`);
      if (contours.length === 0) {
        console.log('No contours found, using default grid');
        // Use centered grid if no contours found
        return [
          [100, 100],
          [924, 100],
          [924, 924],
          [100, 924]
        ];
      }

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
      
      console.log(`Largest contour has area: ${maxArea}, with ${maxContour.length} points`);
      
      // If the largest contour is too small, it's likely noise
      if (maxArea < 10000) {
        console.log('Largest contour too small, likely noise - using default grid');
        return [
          [100, 100],
          [924, 100],
          [924, 924],
          [100, 924]
        ];
      }

      // Approximate corners
      const corners = this.approximateCorners(maxContour);
      
      // Log the corners to help diagnose the issue
      console.log('Raw corners from approximation:', corners);
      
      // Validate the corners - they should form a reasonable quadrilateral
      // Make sure we have 4 corners that form a reasonably sized quadrilateral
      if (corners.length !== 4) {
        console.log(`Expected 4 corners but got ${corners.length}, using default grid`);
        return [
          [100, 100],
          [924, 100],
          [924, 924],
          [100, 924]
        ];
      }
      
      return corners;
    } catch (error) {
      console.error('Error finding corners from mask:', error);
      // Return a default grid if all else fails
      return [
        [100, 100],
        [924, 100],
        [924, 924],
        [100, 924]
      ];
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
      const dist = this.pointToLineDistanceCalculator(points[i], points[0], points[points.length - 1]);
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

  private pointToLineDistanceCalculator(point: number[], lineStart: number[], lineEnd: number[]): number {
    const [x0, y0] = point;
    const [x1, y1] = lineStart;
    const [x2, y2] = lineEnd;

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

  private countTiles(masks: any): { rows: number, cols: number } {
    try {
      // Check if masks is not an array (it could be a tensor)
      if (!Array.isArray(masks)) {
        if (masks && masks.data) {
          // It's a tensor, try to analyze directly
          console.log('Received a tensor for masks, analyzing...');
          
          // Simple method: count regions with mask value > threshold
          let count = 0;
          for (let i = 0; i < masks.data.length; i++) {
            if (masks.data[i] > 0.5) count++;
          }
          
          // Estimate grid size based on count of masked pixels
          console.log(`Found approximately ${count} masked pixels`);
          
          // For simplicity, assume a 4x4 grid
          return { rows: 4, cols: 4 };
        }
        
        console.warn('Invalid masks format:', typeof masks);
        return { rows: 4, cols: 4 }; // Default
      }
      
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

  async processImage(image: HTMLImageElement): Promise<PuzzleState> {
    console.log('Starting image processing');
    this.image = image;
    
    // Try a new hybrid approach - use MobileSAM to identify the puzzle first
    const grid = await this.detectPuzzleWithSAM(image);
    
    if (!grid) {
      console.warn('No valid puzzle grid detected');
      return {
        grid: [[0]],
        size: { rows: 1, cols: 1 },
        emptyPosition: { row: 0, col: 0 },
        pieces: []
      };
    }
    
    console.log(`Detected puzzle grid with size: ${grid.size.rows}x${grid.size.cols}`);
    
    // Extract tile images
    const tileImages = await this.extractTileImages(image, grid);
    console.log(`Extracted ${tileImages.length} tile images`);
    
    // Create a solved grid state based on the detected grid size
    const { rows, cols } = grid.size;
    const gridState: number[][] = [];
    const pieces: PuzzlePiece[] = [];
    
    // Initialize tiles with IDs and images
    let tileId = 1;
    for (let r = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c = 0; c < cols; c++) {
        // If we're at the bottom right, this is the empty tile
        if (r === rows - 1 && c === cols - 1) {
          row.push(0); // Empty tile
        } else {
          row.push(tileId);
          
          // Find the corresponding tile image
          const tileIndex = r * cols + c;
          if (tileIndex < tileImages.length) {
            const tileCanvas = tileImages[tileIndex];
            const tileImage = tileCanvas.toDataURL('image/png');
            
            // Create the piece with correct ID and image
            pieces.push({
              id: tileId,
              currentPosition: { row: r, col: c },
              correctPosition: { row: r, col: c },
              image: tileImage
            });
          } else {
            // No image available for this tile, just add the ID
            pieces.push({
              id: tileId,
              currentPosition: { row: r, col: c },
              correctPosition: { row: r, col: c },
              image: '' // Empty string for image
            });
          }
          
          tileId++;
        }
      }
      gridState.push(row);
    }
    
    // Return the puzzle state with the extracted tiles
    return {
      grid: gridState,
      size: { rows, cols },
      emptyPosition: { row: rows - 1, col: cols - 1 },
      pieces
    };
  }
  
  private async extractTileImages(image: HTMLImageElement, grid: { corners: number[][], size: { rows: number, cols: number } }): Promise<HTMLCanvasElement[]> {
    const { corners, size } = grid;
    const { rows, cols } = size;
    const tileImages: HTMLCanvasElement[] = [];
    
    try {
      // Find the bounds of the grid
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      
      for (const corner of corners) {
        minX = Math.min(minX, corner[0]);
        minY = Math.min(minY, corner[1]);
        maxX = Math.max(maxX, corner[0]);
        maxY = Math.max(maxY, corner[1]);
      }
      
      const gridWidth = maxX - minX;
      const gridHeight = maxY - minY;
      
      console.log(`Grid bounds: (${minX},${minY}) to (${maxX},${maxY}), dimensions: ${gridWidth}x${gridHeight}`);
      
      // Validate grid dimensions again to ensure they're reasonable
      if (gridWidth < 50 || gridHeight < 50) {
        console.warn('Grid dimensions too small for extracting tiles, skipping extraction');
        return [];
      }
      
      // Calculate tile dimensions
      const tileWidth = gridWidth / cols;
      const tileHeight = gridHeight / rows;
      
      // Validate tile dimensions
      if (tileWidth < 10 || tileHeight < 10) {
        console.warn(`Tile dimensions too small (${tileWidth}x${tileHeight}), skipping extraction`);
        return [];
      }
      
      console.log(`Tile dimensions: ${tileWidth}x${tileHeight} pixels`);
      
      // Create a canvas to work with
      const srcCanvas = document.createElement('canvas');
      srcCanvas.width = image.width;
      srcCanvas.height = image.height;
      const srcCtx = srcCanvas.getContext('2d');
      if (!srcCtx) throw new Error('Failed to get canvas context');
      
      // Draw the image at its original size to preserve quality
      srcCtx.drawImage(image, 0, 0, image.width, image.height);
      
      // Create an array to hold all tiles in proper order
      const tilesGrid: HTMLCanvasElement[][] = Array(rows).fill(null).map(() => Array(cols).fill(null));
      
      // Extract each tile with a small margin to avoid cutting at exact edges
      const margin = 0; // No margin to ensure clean tile boundaries
      
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          // Calculate tile position
          const tileX = minX + c * tileWidth;
          const tileY = minY + r * tileHeight;
          
          // Calculate the source rectangle
          const srcX = Math.max(0, tileX - margin);
          const srcY = Math.max(0, tileY - margin);
          const srcWidth = Math.min(tileWidth + 2 * margin, image.width - srcX);
          const srcHeight = Math.min(tileHeight + 2 * margin, image.height - srcY);
          
          // Skip tiles that would be outside the image bounds
          if (srcX >= image.width || srcY >= image.height || srcWidth <= 0 || srcHeight <= 0) {
            console.warn(`Skipping tile at position (${r},${c}) - outside bounds`);
            continue;
          }
          
          // Create a canvas for this tile
          const tileCanvas = document.createElement('canvas');
          tileCanvas.width = 100; // Fixed tile size for display
          tileCanvas.height = 100;
          const tileCtx = tileCanvas.getContext('2d');
          if (!tileCtx) continue;
          
          // Draw the tile portion onto the tile canvas
          tileCtx.drawImage(
            srcCanvas, 
            srcX, srcY, srcWidth, srcHeight, // Source rectangle
            0, 0, 100, 100 // Destination rectangle (fixed size)
          );
          
          // Add a border to make tile boundaries clear
          tileCtx.strokeStyle = 'rgba(255,255,255,0.5)';
          tileCtx.lineWidth = 1;
          tileCtx.strokeRect(0, 0, 100, 100);
          
          // Store in the grid at the correct position
          tilesGrid[r][c] = tileCanvas;
          
          console.log(`Extracted tile at position (${r},${c}): ${srcX},${srcY} with dimensions ${srcWidth}x${srcHeight}`);
        }
      }
      
      // Flatten the grid into a 1D array, going row by row
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (tilesGrid[r][c]) {
            tileImages.push(tilesGrid[r][c]);
          }
        }
      }
      
      console.log(`Extracted and organized ${tileImages.length} tile images in row-by-row order`);
      
      return tileImages;
    } catch (error) {
      console.error('Error extracting tile images:', error);
      return [];
    }
  }
  
  private generatePuzzleState(size: { rows: number, cols: number }, tileImages: HTMLCanvasElement[] = []): PuzzleState {
    console.log('Generating puzzle state for grid of size:', size);
    
    const { rows, cols } = size;
    const grid: number[][] = [];
    const pieces: PuzzlePiece[] = [];
    
    // Create a solved puzzle state and shuffle it
    let counter = 1;
    for (let r = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c = 0; c < cols; c++) {
        // Last tile is empty (0)
        if (r === rows - 1 && c === cols - 1) {
          row.push(0);
        } else {
          row.push(counter++);
        }
        
        // Create a piece with image if available
        if (tileImages.length > r * cols + c) {
          pieces.push({
            id: r * cols + c + 1,
            currentPosition: { row: r, col: c },
            correctPosition: { row: r, col: c },
            image: tileImages[r * cols + c].toDataURL()
          });
        }
      }
      grid.push(row);
    }
    
    // Shuffle the grid 
    this.shufflePuzzle(grid);
    
    // Update the positions of the pieces after shuffling
    if (pieces.length > 0) {
      const piecePositions = new Map<number, { row: number, col: number }>();
      
      // Find the current position of each piece number
      for (let r = 0; r < grid.length; r++) {
        for (let c = 0; c < grid[r].length; c++) {
          const pieceId = grid[r][c];
          if (pieceId !== 0) { // Skip empty space
            piecePositions.set(pieceId, { row: r, col: c });
          }
        }
      }
      
      // Update each piece's position
      for (const piece of pieces) {
        const newPos = piecePositions.get(piece.id);
        if (newPos) {
          piece.currentPosition = newPos;
        }
      }
    }
    
    const result: PuzzleState = {
      grid,
      emptyPosition: this.findEmptyPosition(grid),
      size
    };
    
    // Add pieces property if we have pieces
    if (pieces.length > 0) {
      // @ts-ignore - We're extending the PuzzleState type with pieces
      result.pieces = pieces;
    }
    
    return result;
  }
  
  private shufflePuzzle(grid: number[][]): void {
    const rows = grid.length;
    const cols = grid[0].length;
    const moves = 50; // Number of random moves to make
    
    // Find empty position
    let emptyRow = rows - 1;
    let emptyCol = cols - 1;
    
    for (let i = 0; i < moves; i++) {
      // Determine possible moves (up, down, left, right)
      const possibleMoves = [];
      if (emptyRow > 0) possibleMoves.push({ dr: -1, dc: 0 }); // up
      if (emptyRow < rows - 1) possibleMoves.push({ dr: 1, dc: 0 }); // down
      if (emptyCol > 0) possibleMoves.push({ dr: 0, dc: -1 }); // left
      if (emptyCol < cols - 1) possibleMoves.push({ dr: 0, dc: 1 }); // right
      
      // Choose a random move
      const move = possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
      
      // Swap the empty tile with the chosen adjacent tile
      const newEmptyRow = emptyRow + move.dr;
      const newEmptyCol = emptyCol + move.dc;
      
      grid[emptyRow][emptyCol] = grid[newEmptyRow][newEmptyCol];
      grid[newEmptyRow][newEmptyCol] = 0;
      
      emptyRow = newEmptyRow;
      emptyCol = newEmptyCol;
    }
  }
  
  private findEmptyPosition(grid: number[][]): { row: number, col: number } {
    for (let r = 0; r < grid.length; r++) {
      for (let c = 0; c < grid[r].length; c++) {
        if (grid[r][c] === 0) {
          return { row: r, col: c };
        }
      }
    }
    // Fallback to bottom-right corner if not found
    return { row: grid.length - 1, col: grid[0].length - 1 };
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

      // Process the image and return the puzzle state
      return await this.processImage(image);
    } catch (error) {
      console.error('Error processing frame:', error);
      return null;
    }
  }

  public async detectPuzzleInAR(video: HTMLVideoElement): Promise<PuzzleState | null> {
    console.log('Starting AR puzzle detection...');
    if (!this.isInitialized) {
      console.error('ImageProcessor not initialized');
      return null;
    }

    try {
      // For now, we'll just use the same frame processing logic
      // This will be enhanced with AR-specific detection in the future
      return await this.processFrame(video);
    } catch (error) {
      console.error('Error in AR detection:', error);
      return null;
    }
  }

  private async examineModelStructure(): Promise<void> {
    console.log('Examining model structure...');
    
    // Examine encoder
    if (this.encoderSession) {
      console.log('ENCODER MODEL:');
      try {
        // Try to access session metadata
        // @ts-ignore - May not be supported in all ONNX Runtime versions
        if (this.encoderSession.inputNames) {
          // @ts-ignore
          console.log('Encoder inputs:', this.encoderSession.inputNames);
        } else {
          console.log('Encoder input names not available');
        }
        
        // @ts-ignore
        if (this.encoderSession.outputNames) {
          // @ts-ignore
          console.log('Encoder outputs:', this.encoderSession.outputNames);
        } else {
          console.log('Encoder output names not available');
        }
        
        // Try various common input names with a dummy tensor
        const dummyTensor = new this.ort.Tensor(
          'float32', 
          new Float32Array(3 * 1024 * 1024).fill(0.5),
          [3, 1024, 1024]
        );
        
        const possibleInputNames = [
          'input', 'input_image', 'image', 'images', 
          'encoder_input_image', 'x', 'input_x', 'input_0'
        ];
        
        for (const inputName of possibleInputNames) {
          try {
            console.log(`Trying encoder with input name: ${inputName}`);
            const result = await this.encoderSession.run({ [inputName]: dummyTensor });
            console.log(`Encoder success with input name: ${inputName}`);
            console.log('Encoder outputs:', Object.keys(result));
            console.log('Output shapes:', Object.keys(result).map(key => {
              const tensor = result[key];
              return `${key}: ${JSON.stringify(tensor.dims)}`;
            }));
            break; // Found a working input name
          } catch (error: any) {
            console.log(`Encoder failed with input name ${inputName}:`, error?.message);
          }
        }
      } catch (error) {
        console.error('Error examining encoder structure:', error);
      }
    } else {
      console.log('No encoder session available to examine');
    }
    
    // Examine decoder
    if (!this.session) {
      console.warn('No decoder session available to examine');
      return;
    }
    
    console.log('\nDECODER MODEL:');
    try {
      // Try to access session metadata
      // @ts-ignore - May not be supported in all ONNX Runtime versions
      if (this.session.inputNames) {
        // @ts-ignore
        console.log('Decoder inputs:', this.session.inputNames);
      } else {
        console.log('Decoder input names not available');
      }
      
      // @ts-ignore
      if (this.session.outputNames) {
        // @ts-ignore
        console.log('Decoder outputs:', this.session.outputNames);
      } else {
        console.log('Decoder output names not available');
      }
    } catch (error) {
      console.error('Error examining decoder structure:', error);
    }
  }

  private async logModelDetails(): Promise<void> {
    try {
      console.log('----- MobileSAM Model Details -----');
      
      // Log encoder details if available
      if (this.encoderSession) {
        console.log('ENCODER MODEL:');
        try {
          // @ts-ignore - Access internal properties
          if (this.encoderSession.inputNames) {
            // @ts-ignore
            console.log('Encoder inputs:', this.encoderSession.inputNames);
            // @ts-ignore
            for (const name of this.encoderSession.inputNames) {
              try {
                // @ts-ignore
                const info = this.encoderSession._model.inputs.find((i: any) => i.name === name);
                if (info) {
                  console.log(`  ${name}: type=${info.type}, shape=${JSON.stringify(info.dims)}`);
                }
              } catch (e) {
                console.log(`  ${name}: info not available`);
              }
            }
          }
          
          // @ts-ignore
          if (this.encoderSession.outputNames) {
            // @ts-ignore
            console.log('Encoder outputs:', this.encoderSession.outputNames);
            // @ts-ignore
            for (const name of this.encoderSession.outputNames) {
              try {
                // @ts-ignore
                const info = this.encoderSession._model.outputs.find((o: any) => o.name === name);
                if (info) {
                  console.log(`  ${name}: type=${info.type}, shape=${JSON.stringify(info.dims)}`);
                }
              } catch (e) {
                console.log(`  ${name}: info not available`);
              }
            }
          }
        } catch (error) {
          console.log('Could not access detailed encoder metadata:', error);
        }
      } else {
        console.log('ENCODER: Not available');
      }
      
      // Log decoder details
      console.log('\nDECODER MODEL:');
      if (!this.session) {
        console.warn('No decoder session available');
      } else {
        try {
          // @ts-ignore - Access internal properties
          if (this.session.inputNames) {
            // @ts-ignore
            console.log('Decoder expected inputs:', this.session.inputNames);
            // @ts-ignore
            for (const name of this.session.inputNames) {
              try {
                // @ts-ignore
                const info = this.session._model.inputs.find((i: any) => i.name === name);
                if (info) {
                  console.log(`  ${name}: type=${info.type}, shape=${JSON.stringify(info.dims)}`);
                }
              } catch (e) {
                console.log(`  ${name}: info not available`);
              }
            }
          }
          
          // @ts-ignore
          if (this.session.outputNames) {
            // @ts-ignore
            console.log('Decoder expected outputs:', this.session.outputNames);
            // @ts-ignore
            for (const name of this.session.outputNames) {
              try {
                // @ts-ignore
                const info = this.session._model.outputs.find((o: any) => o.name === name);
                if (info) {
                  console.log(`  ${name}: type=${info.type}, shape=${JSON.stringify(info.dims)}`);
                }
              } catch (e) {
                console.log(`  ${name}: info not available`);
              }
            }
          }
        } catch (error) {
          console.log('Could not access detailed decoder metadata:', error);
        }
      }
      
      console.log('----- End Model Details -----');
    } catch (error) {
      console.error('Error logging model details:', error);
    }
  }

  private async safeLogModelDetails(): Promise<void> {
    console.log('----- MobileSAM Model Details -----');
    
    try {
      // Log encoder metadata safely
      if (this.encoderSession) {
        console.log('ENCODER MODEL:');
        try {
          // @ts-ignore
          console.log('Encoder inputs:', this.encoderSession.inputNames || 'Not available');
          // @ts-ignore
          console.log('Encoder outputs:', this.encoderSession.outputNames || 'Not available');
        } catch (e) {
          console.log('Could not access encoder metadata');
        }
      } else {
        console.log('ENCODER: Not available');
      }
      
      // Log decoder metadata safely
      console.log('\nDECODER MODEL:');
      if (this.session) {
        try {
          // @ts-ignore
          console.log('Decoder inputs:', this.session.inputNames || 'Not available'); 
          // @ts-ignore
          console.log('Decoder outputs:', this.session.outputNames || 'Not available');
        } catch (e) {
          console.log('Could not access decoder metadata');
        }
      } else {
        console.log('DECODER: Not available');
      }
    } catch (error) {
      console.error('Error logging model details:', error);
    }
    
    console.log('----- End Model Details -----');
  }

  /**
   * Specialized detector for slide puzzles using edge detection and grid line analysis
   */
  private async detectSlidePuzzleGrid(image: HTMLImageElement): Promise<{ corners: number[][], size: { rows: number, cols: number } } | null> {
    console.log('Starting specialized slide puzzle detection...');
    try {
      // Create a canvas to draw the image for processing
      const canvas = document.createElement('canvas');
      canvas.width = 800;
      canvas.height = 800;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }
      
      // Scale and draw the image to fill the canvas with black background
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Preserve aspect ratio while filling the canvas
      const scale = Math.max(canvas.width / image.width, canvas.height / image.height);
      const scaledWidth = image.width * scale;
      const scaledHeight = image.height * scale;
      const offsetX = (canvas.width - scaledWidth) / 2;
      const offsetY = (canvas.height - scaledHeight) / 2;
      
      ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);
      
      // Step 1: Find the outer boundary of the puzzle (the frame)
      console.log('Finding outer boundary of the puzzle...');
      const outerBoundary = this.findOuterPuzzleBoundary(canvas);
      
      if (!outerBoundary || outerBoundary.length < 4) {
        console.warn('Could not find valid outer puzzle boundary');
        return null;
      }
      
      // Extract just the puzzle area from the image using the outer boundary
      const outerBounds = this.boundingRect(outerBoundary);
      
      // Add some margin to the bounds to make sure we get the whole frame
      const margin = 5;
      const x = Math.max(0, outerBounds.x - margin);
      const y = Math.max(0, outerBounds.y - margin);
      const width = Math.min(canvas.width - x, outerBounds.width + 2 * margin);
      const height = Math.min(canvas.height - y, outerBounds.height + 2 * margin);
      
      const puzzleCanvas = document.createElement('canvas');
      puzzleCanvas.width = 800;
      puzzleCanvas.height = 800;
      
      const puzzleCtx = puzzleCanvas.getContext('2d');
      if (!puzzleCtx) {
        throw new Error('Failed to get puzzle canvas context');
      }
      
      // Clear with black background
      puzzleCtx.fillStyle = 'black';
      puzzleCtx.fillRect(0, 0, puzzleCanvas.width, puzzleCanvas.height);
      
      // Draw just the puzzle area, scaling it to fill the canvas
      puzzleCtx.drawImage(
        canvas, 
        x, y, width, height,
        0, 0, puzzleCanvas.width, puzzleCanvas.height
      );
      
      console.log(`Isolated puzzle area: ${width}x${height}`);
      
      // Step 2: Try to find an inner boundary within the frame
      // This handles cases where the puzzle has a visible frame around it
      console.log('Looking for inner puzzle grid...');
      
      // Enhance edges to find the inner grid
      const edgeCanvas = this.detectEdges(puzzleCanvas);
      const innerBoundary = this.findInnerPuzzleArea(edgeCanvas, puzzleCanvas.width, puzzleCanvas.height);
      
      // Canvas we'll use for further processing
      let processingCanvas = puzzleCanvas;
      let innerBounds = null;
      
      if (innerBoundary && this.isValidInnerBoundary(innerBoundary, puzzleCanvas.width, puzzleCanvas.height)) {
        // We found a valid inner area within the frame
        console.log('Found inner puzzle area, using that for grid detection');
        
        innerBounds = this.boundingRect(innerBoundary);
        console.log(`Inner bounds: ${innerBounds.width}x${innerBounds.height} at (${innerBounds.x},${innerBounds.y})`);
        
        // Create a new canvas with just the inner area
        const innerCanvas = document.createElement('canvas');
        innerCanvas.width = 800;
        innerCanvas.height = 800;
        
        const innerCtx = innerCanvas.getContext('2d');
        if (!innerCtx) {
          throw new Error('Failed to get inner canvas context');
        }
        
        // Clear canvas
        innerCtx.fillStyle = 'black';
        innerCtx.fillRect(0, 0, innerCanvas.width, innerCanvas.height);
        
        // Add a small margin to ensure we include the grid lines
        const innerMargin = 5;
        innerCtx.drawImage(
          puzzleCanvas,
          Math.max(0, innerBounds.x - innerMargin), 
          Math.max(0, innerBounds.y - innerMargin), 
          Math.min(puzzleCanvas.width - innerBounds.x + innerMargin, innerBounds.width + 2 * innerMargin),
          Math.min(puzzleCanvas.height - innerBounds.y + innerMargin, innerBounds.height + 2 * innerMargin),
          0, 0, innerCanvas.width, innerCanvas.height
        );
        
        processingCanvas = innerCanvas;
      }
      
      // Step 3: Apply edge detection to find the grid lines
      const edgesCanvas = this.detectEdges(processingCanvas);
      
      // Find horizontal and vertical lines
      const { horizontalLines, verticalLines } = this.findGridLines(edgesCanvas);
      console.log(`Detected ${horizontalLines.length} horizontal and ${verticalLines.length} vertical lines`);
      
      // Filter grid lines to find the most regular grid pattern
      let filteredHorizontal = horizontalLines;
      let filteredVertical = verticalLines;
      
      if (horizontalLines.length > 2 && verticalLines.length > 2) {
        filteredHorizontal = this.filterGridLines(horizontalLines, processingCanvas.height);
        filteredVertical = this.filterGridLines(verticalLines, processingCanvas.width);
        console.log(`After filtering: ${filteredHorizontal.length} horizontal and ${filteredVertical.length} vertical lines`);
      }
      
      // Step 4: Determine grid size (rows x cols)
      // For slide puzzles, common sizes are 3x3, 4x4, or 5x5 (with one empty spot)
      // Let's prioritize 4x4 as the default for modern slide puzzles
      let rows = 4;
      let cols = 4;
      
      // If we have enough lines to make a good estimate
      if (filteredHorizontal.length >= 2 && filteredVertical.length >= 2) {
        // First check if we can detect the empty tile
        const emptyTileInfo = this.detectEmptyTile(processingCanvas, filteredHorizontal, filteredVertical);
        
        if (emptyTileInfo.found && emptyTileInfo.estimatedSize) {
          console.log(`Detected empty tile at approximate position: row ${emptyTileInfo.row}, col ${emptyTileInfo.col}`);
          
          // Round to closest standard size
          const estimatedSize = Math.round(emptyTileInfo.estimatedSize);
          
          // Only accept if it's a common puzzle size
          if (estimatedSize >= 3 && estimatedSize <= 5) {
            console.log(`Empty tile suggests a grid size close to ${estimatedSize}x${estimatedSize}`);
            rows = cols = estimatedSize;
          } else {
            // If estimated size is unusual, check which standard size is closest
            const standardSizes = [3, 4, 5];
            const closestSize = standardSizes.reduce((prev, curr) => 
              Math.abs(curr - emptyTileInfo.estimatedSize!) < Math.abs(prev - emptyTileInfo.estimatedSize!)
                ? curr : prev
            );
            console.log(`Estimated size ${estimatedSize} unusual, using closest standard size: ${closestSize}x${closestSize}`);
            rows = cols = closestSize;
          }
        } else {
          // If we can't find the empty tile, evaluate the number of lines
          // For an NxN grid, we expect N-1 inner lines plus 2 boundary lines = N+1 lines
          const estimatedRows = filteredHorizontal.length - 1;
          const estimatedCols = filteredVertical.length - 1;
          
          // Take the most confident dimension (the one with more lines)
          let estimatedSize = Math.max(estimatedRows, estimatedCols);
          
          // Make sure it's a common puzzle size
          if (estimatedSize < 3) estimatedSize = 3;
          if (estimatedSize > 5) estimatedSize = 5;
          
          console.log(`Based on line count, estimated grid size: ${estimatedSize}x${estimatedSize}`);
          rows = cols = estimatedSize;
        }
      } else {
        // Not enough lines - just assume it's a standard 4x4 puzzle
        console.log('Not enough lines detected, defaulting to 4x4 grid');
      }
      
      console.log(`Final grid dimensions: ${rows}x${cols}`);
      
      // Step 5: Generate grid corners
      // We need (rows+1) x (cols+1) points forming the corners of the grid
      const cornerPoints: number[][] = [];
      
      // Use the full canvas dimensions to generate a regular grid
      // This works whether we're using the whole puzzle or just the inner area
      for (let r = 0; r <= rows; r++) {
        for (let c = 0; c <= cols; c++) {
          cornerPoints.push([
            c * processingCanvas.width / cols,
            r * processingCanvas.height / rows
          ]);
        }
      }
      
      console.log(`Generated ${cornerPoints.length} grid corner points`);
      
      // Step 6: Scale corners back to original image coordinates
      const scaledCorners = cornerPoints.map(corner => {
        let [x, y] = corner;
        
        // If we found an inner boundary, adjust points relative to it
        if (innerBounds) {
          // Map from processing canvas back to the inner boundary within the puzzle canvas
          x = x * innerBounds.width / processingCanvas.width + innerBounds.x;
          y = y * innerBounds.height / processingCanvas.height + innerBounds.y;
        }
        
        // Then map from puzzle canvas to the outer boundary within the main canvas
        x = x * width / puzzleCanvas.width + x;
        y = y * height / puzzleCanvas.height + y;
        
        // Finally map from main canvas to original image
        x = (x - offsetX) / scale;
        y = (y - offsetY) / scale;
        
        return [x, y];
      });
      
      return {
        corners: scaledCorners,
        size: { rows, cols }
      };
    } catch (error) {
      console.error('Error in slide puzzle detection:', error);
      return null;
    }
  }
  
  // Helper method to find an inner boundary within the puzzle frame
  private findInnerPuzzleArea(edgeCanvas: HTMLCanvasElement, width: number, height: number): number[][] | null {
    try {
      console.log('Searching for inner puzzle area within frame...');
      
      // Find contours in the edge detection image
      const contours = this.findContours(edgeCanvas);
      if (!contours || contours.length === 0) {
        console.log('No contours found in inner area search');
        return null;
      }
      
      console.log(`Found ${contours.length} inner contours`);
      
      // Filter contours by size and shape
      // We're looking for a large rectangular contour that's smaller than the full canvas
      const validContours = contours
        .filter(contour => {
          const area = this.calculateContourArea(contour);
          if (area < 5000) {
            return false; // Too small
          }
          
          const rect = this.boundingRect(contour);
          const rectArea = rect.width * rect.height;
          
          // Check if the contour covers a significant portion but not all of the image
          const canvasArea = width * height;
          const coverageRatio = rectArea / canvasArea;
          
          if (coverageRatio < 0.2 || coverageRatio > 0.9) {
            return false; // Either too small or too large relative to the canvas
          }
          
          // Check if it's reasonably rectangular
          const rectangularity = Math.min(area / rectArea, 1);
          const isRectangular = rectangularity > 0.7;
          
          // Check if it's centered enough
          const centerX = rect.x + rect.width / 2;
          const centerY = rect.y + rect.height / 2;
          const normalizedDistanceFromCenter = Math.sqrt(
            Math.pow((centerX - width/2) / width, 2) + 
            Math.pow((centerY - height/2) / height, 2)
          );
          
          const isCentered = normalizedDistanceFromCenter < 0.2;
          
          console.log(`Contour: area=${area}, ratio=${coverageRatio.toFixed(2)}, rect=${rectangularity.toFixed(2)}, centered=${isCentered}`);
          
          return isRectangular && isCentered;
        })
        .sort((a, b) => {
          // Prefer contours that are more square-like (aspect ratio close to 1)
          const rectA = this.boundingRect(a);
          const rectB = this.boundingRect(b);
          
          const aspectRatioA = Math.max(rectA.width / rectA.height, rectA.height / rectA.width);
          const aspectRatioB = Math.max(rectB.width / rectB.height, rectB.height / rectB.width);
          
          // Prefer aspect ratios closer to 1 (square)
          const aspectScore = aspectRatioA - aspectRatioB;
          
          // If aspect ratios are similar, prefer larger contours
          if (Math.abs(aspectScore) < 0.1) {
            const areaA = this.calculateContourArea(a);
            const areaB = this.calculateContourArea(b);
            return areaB - areaA; // Larger area first
          }
          
          return aspectScore; // Closer to square first
        });
      
      if (validContours.length === 0) {
        console.log('No valid inner contours found');
        return null;
      }
      
      // Get the best contour and approximate it
      const bestContour = validContours[0];
      const rect = this.boundingRect(bestContour);
      console.log(`Found inner puzzle area: ${rect.width}x${rect.height} at (${rect.x},${rect.y})`);
      
      // Return a rectangle based on the bounding box
      // This ensures we have a perfect rectangle for the puzzle grid
      return [
        [rect.x, rect.y],
        [rect.x + rect.width, rect.y],
        [rect.x + rect.width, rect.y + rect.height],
        [rect.x, rect.y + rect.height]
      ];
    } catch (error) {
      console.error('Error finding inner puzzle area:', error);
      return null;
    }
  }
  
  // Check if the inner boundary we found is valid
  private isValidInnerBoundary(boundary: number[][], width: number, height: number): boolean {
    if (!boundary || boundary.length < 4) {
      return false;
    }
    
    // Check if the boundary is reasonably rectangular
    const bounds = this.boundingRect(boundary);
    const area = this.calculateContourArea(boundary);
    const rectArea = bounds.width * bounds.height;
    const rectangularity = Math.min(area / rectArea, 1);
    
    if (rectangularity < 0.75) {
      return false; // Not rectangular enough
    }
    
    // Check if it covers a reasonable portion of the image
    const canvasArea = width * height;
    const coverageRatio = rectArea / canvasArea;
    
    if (coverageRatio < 0.25 || coverageRatio > 0.95) {
      return false; // Either too small or too large
    }
    
    // Check if it's somewhat centered in the image
    const centerX = width / 2;
    const centerY = height / 2;
    
    const boundaryCenter = {
      x: bounds.x + bounds.width / 2,
      y: bounds.y + bounds.height / 2
    };
    
    const normalizedDistance = Math.sqrt(
      Math.pow((boundaryCenter.x - centerX) / width, 2) + 
      Math.pow((boundaryCenter.y - centerY) / height, 2)
    );
    
    if (normalizedDistance > 0.2) {
      return false; // Too far from center
    }
    
    return true;
  }
  
  // Detect an empty tile in the puzzle to help estimate grid size
  private detectEmptyTile(canvas: HTMLCanvasElement, horizontalLines: number[], verticalLines: number[]): { 
    found: boolean, 
    row?: number, 
    col?: number, 
    estimatedSize?: number 
  } {
    try {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        return { found: false };
      }
      
      // Sort lines
      horizontalLines.sort((a, b) => a - b);
      verticalLines.sort((a, b) => a - b);
      
      // Add canvas bounds
      const allHorizontal = [0, ...horizontalLines, canvas.height];
      const allVertical = [0, ...verticalLines, canvas.width];
      
      // Calculate the average cell size
      let totalCellWidth = 0;
      let totalCellHeight = 0;
      let cellWidthCount = 0;
      let cellHeightCount = 0;
      
      for (let i = 1; i < allVertical.length; i++) {
        const width = allVertical[i] - allVertical[i-1];
        if (width > 10) { // Ignore tiny gaps
          totalCellWidth += width;
          cellWidthCount++;
        }
      }
      
      for (let i = 1; i < allHorizontal.length; i++) {
        const height = allHorizontal[i] - allHorizontal[i-1];
        if (height > 10) { // Ignore tiny gaps
          totalCellHeight += height;
          cellHeightCount++;
        }
      }
      
      const avgCellWidth = cellWidthCount > 0 ? totalCellWidth / cellWidthCount : 0;
      const avgCellHeight = cellHeightCount > 0 ? totalCellHeight / cellHeightCount : 0;
      
      // Check each cell for darkness (empty cell is often darker)
      let darkestCell = { row: -1, col: -1, darkness: -1 };
      
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      for (let i = 0; i < allHorizontal.length - 1; i++) {
        for (let j = 0; j < allVertical.length - 1; j++) {
          const x = allVertical[j];
          const y = allHorizontal[i];
          const width = allVertical[j+1] - x;
          const height = allHorizontal[i+1] - y;
          
          // Skip if cell is too small
          if (width < 20 || height < 20) continue;
          
          // Sample the center of the cell
          const centerX = Math.floor(x + width / 2);
          const centerY = Math.floor(y + height / 2);
          
          // Calculate average brightness in a small region around the center
          let totalBrightness = 0;
          let pixelCount = 0;
          
          const sampleSize = 5; // 5x5 sample region
          for (let sy = -sampleSize; sy <= sampleSize; sy++) {
            for (let sx = -sampleSize; sx <= sampleSize; sx++) {
              const sampleX = centerX + sx;
              const sampleY = centerY + sy;
              
              if (sampleX >= 0 && sampleX < canvas.width && sampleY >= 0 && sampleY < canvas.height) {
                const idx = (sampleY * canvas.width + sampleX) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                
                // Calculate brightness (0-255)
                const brightness = (r + g + b) / 3;
                totalBrightness += brightness;
                pixelCount++;
              }
            }
          }
          
          if (pixelCount > 0) {
            const avgBrightness = totalBrightness / pixelCount;
            const darkness = 255 - avgBrightness;
            
            if (darkness > darkestCell.darkness) {
              darkestCell = { row: i, col: j, darkness: darkness };
            }
          }
        }
      }
      
      // Calculate how different the darkest cell is from the average
      if (darkestCell.row >= 0 && darkestCell.darkness > 100) {
        console.log(`Potential empty tile found at row ${darkestCell.row}, col ${darkestCell.col} with darkness ${darkestCell.darkness}`);
        
        // Estimate grid size based on cell dimensions
        const estimatedSize = Math.round(Math.sqrt(
          (canvas.width / avgCellWidth) * (canvas.height / avgCellHeight)
        ));
        
        return { 
          found: true, 
          row: darkestCell.row, 
          col: darkestCell.col,
          estimatedSize
        };
      }
      
      return { found: false };
    } catch (error) {
      console.error('Error detecting empty tile:', error);
      return { found: false };
    }
  }

  /**
   * Find the outer boundary of a puzzle in the image
   */
  private findOuterPuzzleBoundary(canvas: HTMLCanvasElement): number[][] {
    try {
      console.log('Looking for puzzle frame...');
      
      // Create a copy of the canvas for preprocessing
      const preCanvas = document.createElement('canvas');
      preCanvas.width = canvas.width;
      preCanvas.height = canvas.height;
      const preCtx = preCanvas.getContext('2d', { willReadFrequently: true });
      if (!preCtx) {
        console.warn('Failed to get preprocessing canvas context');
        return [];
      }
      
      // Draw the image and enhance edges
      preCtx.drawImage(canvas, 0, 0);
      
      // Apply a threshold to enhance contrast - makes the puzzle frame more obvious
      const imageData = preCtx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      // Calculate the average brightness to adaptively set threshold
      let totalBrightness = 0;
      for (let i = 0; i < data.length; i += 4) {
        totalBrightness += (data[i] + data[i+1] + data[i+2]) / 3;
      }
      const avgBrightness = totalBrightness / (data.length / 4);
      const threshold = avgBrightness * 0.7; // Set threshold relative to average brightness
      
      // Apply threshold
      for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
        const value = brightness > threshold ? 255 : 0;
        data[i] = data[i+1] = data[i+2] = value;
      }
      
      preCtx.putImageData(imageData, 0, 0);
      
      // Find contours in the preprocessed image
      console.log('Finding contours in thresholded image...');
      const contours = this.findContours(preCanvas);
      
      if (!contours || contours.length === 0) {
        console.warn('No contours found in thresholded image');
        
        // Fall back to finding contours in the original
        console.log('Falling back to original image contours...');
        const originalContours = this.findContours(canvas);
        if (!originalContours || originalContours.length === 0) {
          console.warn('No contours found in original image either');
          return [];
        }
        
        contours.push(...originalContours);
      }
      
      console.log(`Found ${contours.length} contours`);
      
      // Filter and score contours for finding a puzzle frame
      // We want a large, rectangular contour near the center of the image
      
      // First, get 5 largest contours by area
      const largeContours = [...contours]
        .filter(contour => this.calculateContourArea(contour) > 5000)
        .sort((a, b) => this.calculateContourArea(b) - this.calculateContourArea(a))
        .slice(0, 5);
      
      if (largeContours.length === 0) {
        console.warn('No large contours found');
        return [];
      }
      
      console.log(`Analyzing ${largeContours.length} largest contours`);
      
      // Score contours based on how likely they are to be a puzzle frame
      let bestContour: number[][] | null = null;
      let bestScore = -1;
      
      for (const contour of largeContours) {
        // Calculate metrics
        const area = this.calculateContourArea(contour);
        const boundingRect = this.boundingRect(contour);
        const rectArea = boundingRect.width * boundingRect.height;
        
        // Check if width and height are similar (puzzle frames are usually square-ish)
        const aspectRatio = Math.max(boundingRect.width / boundingRect.height, boundingRect.height / boundingRect.width);
        
        // Perfect square has aspect ratio 1, penalize as it deviates
        const aspectScore = 1 / (aspectRatio * aspectRatio);
        
        // Check rectangularity - how well the contour fills its bounding rectangle
        const rectangularity = Math.min(area / rectArea, 1.0);
        
        // Calculate center position and how close it is to image center
        const centerX = boundingRect.x + boundingRect.width / 2;
        const centerY = boundingRect.y + boundingRect.height / 2;
        const imageWidth = canvas.width;
        const imageHeight = canvas.height;
        
        // Distance from image center, normalized to [0,1] where 0 is centered
        const distanceFromCenter = Math.sqrt(
          Math.pow((centerX - imageWidth/2) / (imageWidth/2), 2) + 
          Math.pow((centerY - imageHeight/2) / (imageHeight/2), 2)
        );
        const centerScore = 1 - distanceFromCenter;
        
        // Size score - we want it to be a significant portion of the image but not too big
        const relativeSizePercent = rectArea / (imageWidth * imageHeight);
        let sizeScore;
        
        if (relativeSizePercent < 0.1) {
          // Too small, penalize heavily
          sizeScore = relativeSizePercent * 2;
        } else if (relativeSizePercent > 0.9) {
          // Too large, penalize heavily
          sizeScore = (1 - relativeSizePercent) * 2;
        } else {
          // Good size range, score peaks at around 40-60% of image
          sizeScore = 1 - Math.abs(0.5 - relativeSizePercent) * 2;
        }
        
        // Combine scores with different weights
        const scoreWeights = {
          rectangularity: 0.4,   // How rectangular
          aspectRatio: 0.3,      // How square-like
          center: 0.2,           // How centered
          size: 0.1              // Appropriate size
        };
        
        const score = (
          rectangularity * scoreWeights.rectangularity + 
          aspectScore * scoreWeights.aspectRatio + 
          centerScore * scoreWeights.center + 
          sizeScore * scoreWeights.size
        ) * area; // Still favor larger areas when scores are similar
        
        console.log(`Contour: area=${area.toFixed(0)}, rect=${rectangularity.toFixed(2)}, aspect=${aspectScore.toFixed(2)}, center=${centerScore.toFixed(2)}, size=${sizeScore.toFixed(2)}, score=${score.toFixed(0)}`);
        
        if (score > bestScore) {
          bestScore = score;
          bestContour = contour;
        }
      }
      
      if (!bestContour) {
        console.warn('Could not find a suitable puzzle frame');
        return [];
      }
      
      // Get the exact corners from the best contour
      // Rather than approximating, we'll create an exact rectangle from the bounding box
      const rect = this.boundingRect(bestContour);
      console.log(`Selected frame: ${rect.width}x${rect.height} at (${rect.x},${rect.y}), score=${bestScore.toFixed(0)}`);
      
      return [
        [rect.x, rect.y],                     // top-left
        [rect.x + rect.width, rect.y],        // top-right
        [rect.x + rect.width, rect.y + rect.height], // bottom-right
        [rect.x, rect.y + rect.height]        // bottom-left
      ];
    } catch (error) {
      console.error('Error finding outer puzzle boundary:', error);
      return [];
    }
  }
  
  /**
   * Calculate the arc length of a contour (perimeter)
   */
  private arcLength(contour: number[][]): number {
    let length = 0;
    
    for (let i = 0; i < contour.length; i++) {
      const p1 = contour[i];
      const p2 = contour[(i + 1) % contour.length];
      length += Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
    }
    
    return length;
  }
  
  /**
   * Calculate a bounding rectangle for a set of points
   */
  private boundingRect(points: number[][]): { x: number, y: number, width: number, height: number } {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    for (const [x, y] of points) {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    
    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY
    };
  }
  
  /**
   * Apply Canny filter to enhance edges (simplified version)
   */
  private applyCannyFilter(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Simple implementation: just apply a threshold to existing edges
    // A full Canny filter would include gaussian blur, gradient calculation, non-maximum suppression, and hysteresis thresholding
    for (let i = 0; i < data.length; i += 4) {
      // If it's already an edge pixel, keep it, otherwise make it black
      data[i] = data[i] > 200 ? 255 : 0;      // R
      data[i + 1] = data[i + 1] > 200 ? 255 : 0; // G
      data[i + 2] = data[i + 2] > 200 ? 255 : 0; // B
    }
    
    ctx.putImageData(imageData, 0, 0);
  }
  
  /**
   * Find grid lines in an edge-detected image using a Hough-transform inspired approach
   */
  private findGridLines(edgeCanvas: HTMLCanvasElement): { horizontalLines: number[], verticalLines: number[] } {
    console.log('Finding grid lines...');
    const width = edgeCanvas.width;
    const height = edgeCanvas.height;
    
    // Get the edge data
    const ctx = edgeCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      return { horizontalLines: [], verticalLines: [] };
    }
    
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // Project edges onto horizontal and vertical axes
    const horizontalProjection = new Array(height).fill(0);
    const verticalProjection = new Array(width).fill(0);
    
    // Count white pixels (edges) for each row and column
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        if (data[idx] > 128) { // If pixel is bright (edge)
          horizontalProjection[y]++;
          verticalProjection[x]++;
        }
      }
    }
    
    // Smooth the projections to reduce noise
    const smoothProjection = (projection: number[], windowSize: number = 5): number[] => {
      const result = new Array(projection.length).fill(0);
      const halfWindow = Math.floor(windowSize / 2);
      
      for (let i = 0; i < projection.length; i++) {
        let sum = 0;
        let count = 0;
        
        for (let j = Math.max(0, i - halfWindow); j <= Math.min(projection.length - 1, i + halfWindow); j++) {
          sum += projection[j];
          count++;
        }
        
        result[i] = sum / count;
      }
      
      return result;
    };
    
    // Normalize projection to range [0, 1]
    const normalizeProjection = (projection: number[]): number[] => {
      const max = Math.max(...projection);
      if (max === 0) return projection;
      
      return projection.map(v => v / max);
    };
    
    // Find peaks in the projection that likely correspond to grid lines
    const findPeaks = (projection: number[], minDistance: number = 20, threshold: number = 0.2): number[] => {
      const peaks: number[] = [];
      const normalized = normalizeProjection(projection);
      
      // Add a condition for minimum peak height
      const minPeakHeight = threshold;
      
      for (let i = 1; i < normalized.length - 1; i++) {
        // Check if this is a local maximum
        if (normalized[i] > normalized[i - 1] && normalized[i] > normalized[i + 1]) {
          // Check if it's above threshold
          if (normalized[i] > minPeakHeight) {
            // Check if it's far enough from other peaks
            const farEnough = peaks.every(peak => Math.abs(peak - i) >= minDistance);
            
            if (farEnough) {
              peaks.push(i);
            } else {
              // If too close to another peak, keep only the higher one
              const nearestPeak = peaks.reduce((nearest, peak) => {
                return Math.abs(peak - i) < Math.abs(nearest - i) ? peak : nearest;
              }, Infinity);
              
              if (nearestPeak !== Infinity && normalized[i] > normalized[nearestPeak]) {
                // Replace the existing peak with this one
                peaks[peaks.indexOf(nearestPeak)] = i;
              }
            }
          }
        }
      }
      
      return peaks.sort((a, b) => a - b);
    };
    
    // Process the projections
    const smoothedHorizontal = smoothProjection(horizontalProjection);
    const smoothedVertical = smoothProjection(verticalProjection);
    
    // Start with a relatively low threshold to find as many potential lines as possible
    let threshold = 0.2;
    const minLineDistance = Math.min(width, height) * 0.05; // 5% of dimension
    
    // Try different thresholds until we get some lines
    let horizontalLines: number[] = [];
    let verticalLines: number[] = [];
    
    while (threshold > 0.05) {
      horizontalLines = findPeaks(smoothedHorizontal, minLineDistance, threshold);
      verticalLines = findPeaks(smoothedVertical, minLineDistance, threshold);
      
      console.log(`Found ${horizontalLines.length} horizontal lines with threshold ${threshold}`);
      console.log(`Found ${verticalLines.length} vertical lines with threshold ${threshold}`);
      
      // If we found a reasonable number of lines, break
      if (horizontalLines.length >= 3 && verticalLines.length >= 3) {
        console.log('Found enough lines, stopping search');
        break;
      }
      
      // If not enough lines, lower the threshold and try again
      threshold -= 0.05;
    }
    
    // Visualize the lines (useful for debugging)
    try {
      const visualCanvas = document.createElement('canvas');
      visualCanvas.width = width;
      visualCanvas.height = height;
      const visualCtx = visualCanvas.getContext('2d');
      
      if (visualCtx) {
        // Draw the edge image
        visualCtx.drawImage(edgeCanvas, 0, 0);
        
        // Draw horizontal lines
        visualCtx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        visualCtx.lineWidth = 2;
        for (const y of horizontalLines) {
          visualCtx.beginPath();
          visualCtx.moveTo(0, y);
          visualCtx.lineTo(width, y);
          visualCtx.stroke();
        }
        
        // Draw vertical lines
        visualCtx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        for (const x of verticalLines) {
          visualCtx.beginPath();
          visualCtx.moveTo(x, 0);
          visualCtx.lineTo(x, height);
          visualCtx.stroke();
        }
        
        // Draw the projection values
        const normalizedH = normalizeProjection(smoothedHorizontal);
        const normalizedV = normalizeProjection(smoothedVertical);
        
        visualCtx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
        visualCtx.lineWidth = 1;
        
        // Horizontal projection
        visualCtx.beginPath();
        for (let y = 0; y < height; y++) {
          const x = Math.floor(normalizedH[y] * width * 0.2); // Scale for visibility
          if (y === 0) {
            visualCtx.moveTo(width - x, y);
          } else {
            visualCtx.lineTo(width - x, y);
          }
        }
        visualCtx.stroke();
        
        // Vertical projection
        visualCtx.beginPath();
        for (let x = 0; x < width; x++) {
          const y = Math.floor(normalizedV[x] * height * 0.2); // Scale for visibility
          if (x === 0) {
            visualCtx.moveTo(x, height - y);
          } else {
            visualCtx.lineTo(x, height - y);
          }
        }
        visualCtx.stroke();
        
        // Create a data URL for the visualization
        // This could be sent to the frontend for debugging if needed
        const dataURL = visualCanvas.toDataURL();
        console.log('Line detection visualization available');
      }
    } catch (error) {
      console.error('Error creating visualization:', error);
    }
    
    return {
      horizontalLines,
      verticalLines
    };
  }
  
  /**
   * Filter grid lines to find evenly spaced lines that form a grid
   */
  private filterGridLines(lines: number[], maxSize: number): number[] {
    if (lines.length <= 3) return lines; // If we only have a few lines, keep them all
    
    // Calculate spacings between adjacent lines
    const spacings: number[] = [];
    for (let i = 1; i < lines.length; i++) {
      spacings.push(lines[i] - lines[i - 1]);
    }
    
    // Find the most common spacing (for a regular grid)
    const spacingCounts = new Map<number, number>();
    let maxCount = 0;
    let mostCommonSpacing = 0;
    
    // Group similar spacings (within 10% of each other)
    for (const spacing of spacings) {
      let matched = false;
      for (const [existingSpacing, count] of spacingCounts.entries()) {
        if (Math.abs(spacing - existingSpacing) / existingSpacing < 0.1) {
          // Update the existing spacing with the average
          const newSpacing = (existingSpacing * count + spacing) / (count + 1);
          spacingCounts.set(newSpacing, count + 1);
          if (count + 1 > maxCount) {
            maxCount = count + 1;
            mostCommonSpacing = newSpacing;
          }
          spacingCounts.delete(existingSpacing);
          matched = true;
          break;
        }
      }
      
      if (!matched) {
        spacingCounts.set(spacing, 1);
        if (1 > maxCount) {
          maxCount = 1;
          mostCommonSpacing = spacing;
        }
      }
    }
    
    console.log(`Most common spacing: ${mostCommonSpacing} (found ${maxCount} times)`);
    
    // Filter lines to keep only those that are close to the expected spacing
    const filteredLines: number[] = [];
    let expectedPosition = lines[0]; // Start with the first line
    
    filteredLines.push(lines[0]); // Always keep the first line
    
    for (let i = 1; i < lines.length; i++) {
      expectedPosition += mostCommonSpacing;
      
      // Find the line closest to the expected position
      let closestLine = lines[i];
      let minDifference = Math.abs(lines[i] - expectedPosition);
      
      for (let j = i + 1; j < lines.length && j < i + 3; j++) {
        const difference = Math.abs(lines[j] - expectedPosition);
        if (difference < minDifference) {
          minDifference = difference;
          closestLine = lines[j];
        }
      }
      
      // Only add the line if it's close enough to the expected position
      if (minDifference / mostCommonSpacing < 0.2) {
        filteredLines.push(closestLine);
        expectedPosition = closestLine; // Adjust the expected position for the next line
      }
    }
    
    // If we've filtered out too many lines, add some back
    if (filteredLines.length < 3 && lines.length >= 3) {
      return [lines[0], ...lines.slice(Math.max(1, lines.length - 2))];
    }
    
    return filteredLines;
  }

  /**
   * Apply edge detection to find grid lines
   */
  private detectEdges(sourceCanvas: HTMLCanvasElement): HTMLCanvasElement {
    try {
      // Create a new canvas for edge detection
      const canvas = document.createElement('canvas');
      canvas.width = sourceCanvas.width;
      canvas.height = sourceCanvas.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('Failed to get canvas context');
      
      // Draw the source image
      ctx.drawImage(sourceCanvas, 0, 0);
      
      // Get image data for processing
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      // Use a 3x3 Sobel operator for edge detection
      const sobel = new Uint8ClampedArray(data.length);
      
      // Increase sensitivity to detect more edges
      const sensitivity = 1.5; // Increase this for more sensitivity
      
      // Apply Sobel operator
      for (let y = 1; y < canvas.height - 1; y++) {
        for (let x = 1; x < canvas.width - 1; x++) {
          // Pixel positions for 3x3 grid
          const i00 = ((y - 1) * canvas.width + (x - 1)) * 4; // top-left
          const i01 = ((y - 1) * canvas.width + x) * 4;       // top-center
          const i02 = ((y - 1) * canvas.width + (x + 1)) * 4; // top-right
          const i10 = (y * canvas.width + (x - 1)) * 4;       // mid-left
          const i12 = (y * canvas.width + (x + 1)) * 4;       // mid-right
          const i20 = ((y + 1) * canvas.width + (x - 1)) * 4; // bottom-left
          const i21 = ((y + 1) * canvas.width + x) * 4;       // bottom-center
          const i22 = ((y + 1) * canvas.width + (x + 1)) * 4; // bottom-right
          
          // Sobel kernels
          // Horizontal
          const gx = 
            -1 * (data[i00] + data[i00+1] + data[i00+2]) +
            -2 * (data[i10] + data[i10+1] + data[i10+2]) +
            -1 * (data[i20] + data[i20+1] + data[i20+2]) +
            1 * (data[i02] + data[i02+1] + data[i02+2]) +
            2 * (data[i12] + data[i12+1] + data[i12+2]) +
            1 * (data[i22] + data[i22+1] + data[i22+2]);
          
          // Vertical
          const gy = 
            -1 * (data[i00] + data[i00+1] + data[i00+2]) +
            -2 * (data[i01] + data[i01+1] + data[i01+2]) +
            -1 * (data[i02] + data[i02+1] + data[i02+2]) +
            1 * (data[i20] + data[i20+1] + data[i20+2]) +
            2 * (data[i21] + data[i21+1] + data[i21+2]) +
            1 * (data[i22] + data[i22+1] + data[i22+2]);
          
          // Gradient magnitude
          const mag = Math.sqrt(gx * gx + gy * gy) * sensitivity;
          
          // Current pixel position
          const i = (y * canvas.width + x) * 4;
          
          // Strong edges as white, everything else black
          const value = Math.min(255, Math.max(0, mag));
          sobel[i] = sobel[i+1] = sobel[i+2] = value;
          sobel[i+3] = 255; // Alpha
        }
      }
      
      // Apply a threshold to create a binary edge image
      const threshold = 50; // Adjust this for detection sensitivity
      for (let i = 0; i < sobel.length; i += 4) {
        if (sobel[i] > threshold) {
          sobel[i] = sobel[i+1] = sobel[i+2] = 255;
        } else {
          sobel[i] = sobel[i+1] = sobel[i+2] = 0;
        }
      }
      
      // Put the edge-detected data back into the canvas
      const edgeImageData = new ImageData(sobel, canvas.width, canvas.height);
      ctx.putImageData(edgeImageData, 0, 0);
      
      // Apply closing morphological operation to connect broken edges
      // This is done by first dilating then eroding
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) throw new Error('Failed to get temporary canvas context');
      
      // Dilate (expand white regions)
      tempCtx.drawImage(canvas, 0, 0);
      const dilatedData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
      const dilatedPixels = dilatedData.data;
      
      // Use a 3x3 kernel for dilation
      const kernelSize = 1; // 3x3 kernel
      for (let y = kernelSize; y < canvas.height - kernelSize; y++) {
        for (let x = kernelSize; x < canvas.width - kernelSize; x++) {
          let maxVal = 0;
          
          // Check 3x3 neighborhood
          for (let ky = -kernelSize; ky <= kernelSize; ky++) {
            for (let kx = -kernelSize; kx <= kernelSize; kx++) {
              const idx = ((y + ky) * canvas.width + (x + kx)) * 4;
              maxVal = Math.max(maxVal, sobel[idx]);
            }
          }
          
          const idx = (y * canvas.width + x) * 4;
          dilatedPixels[idx] = dilatedPixels[idx+1] = dilatedPixels[idx+2] = maxVal;
        }
      }
      
      tempCtx.putImageData(dilatedData, 0, 0);
      
      // Erode (shrink white regions) - this completes the closing operation
      ctx.drawImage(tempCanvas, 0, 0);
      
      console.log('Edge detection completed');
      return canvas;
    } catch (error) {
      console.error('Error in edge detection:', error);
      return sourceCanvas; // Return original if edge detection fails
    }
  }

  /**
   * New approach: First use MobileSAM to identify the puzzle object,
   * then do grid detection within it
   */
  private async detectPuzzleWithSAM(image: HTMLImageElement): Promise<{ corners: number[][], size: { rows: number, cols: number } } | null> {
    console.log('Starting puzzle detection with MobileSAM...');
    
    try {
      // First make sure MobileSAM is initialized
      if (!this.encoderSession || !this.session) {
        console.error('MobileSAM not initialized');
        return null;
      }
      
      // Step 1: Create a canvas with the image
      const canvas = document.createElement('canvas');
      canvas.width = 800;  // Use a consistent size for processing
      canvas.height = 800;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }
      
      // Draw the image preserving aspect ratio
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
      const scaledWidth = image.width * scale;
      const scaledHeight = image.height * scale;
      const offsetX = (canvas.width - scaledWidth) / 2;
      const offsetY = (canvas.height - scaledHeight) / 2;
      
      ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);
      
      // Convert canvas to image for preprocessImage
      const canvasImage = new Image();
      await new Promise<void>((resolve, reject) => {
        canvasImage.onload = () => resolve();
        canvasImage.onerror = () => reject(new Error('Failed to create image from canvas'));
        canvasImage.src = canvas.toDataURL('image/png');
      });
      
      // Step 2: Preprocess the image for MobileSAM
      console.log('Preprocessing image for MobileSAM...');
      const tensor = await this.preprocessImage(canvasImage);
      
      // Step 3: Generate embeddings
      console.log('Generating image embeddings...');
      const embeddings = await this.generateEmbeddings(tensor);
      
      // If SAM didn't work, fall back to traditional detection
      if (!embeddings) {
        console.log('SAM embeddings generation failed, falling back to traditional detection');
        return this.detectSlidePuzzleGrid(image);
      }
      
      // Step 4: Generate prompt points to find the puzzle object
      // We'll place points in a grid pattern covering more of the image
      const points: number[][] = [];
      
      // Place a 3x3 grid of points across the image
      for (let y = 0.25; y <= 0.75; y += 0.25) {
        for (let x = 0.25; x <= 0.75; x += 0.25) {
          points.push([x * canvas.width, y * canvas.height]);
        }
      }
      
      console.log(`Created ${points.length} prompt points for object detection`);
      
      // Step 5: Generate mask using points
      console.log('Generating masks with prompt points...');
      const masks = await this.generateMask(embeddings, points, true);
      
      if (!masks || !masks.length) {
        console.warn('No masks generated from prompt points, falling back to traditional detection');
        return this.detectSlidePuzzleGrid(image);
      }
      
      console.log(`Generated ${masks.length} potential object masks`);
      
      // Step 6: Find the best mask (largest connected area in the center)
      let bestMask = null;
      let bestScore = -1;
      
      for (let i = 0; i < masks.length; i++) {
        const mask = masks[i];
        const area = this.calculateMaskArea(mask);
        const centerDistance = this.calculateMaskCenterDistance(mask, canvas.width, canvas.height);
        const score = area * (1 - centerDistance * 2);  // Prioritize large, centered masks
        
        console.log(`Mask ${i}: area=${area.toFixed(0)}, centerDist=${centerDistance.toFixed(2)}, score=${score.toFixed(0)}`);
        
        if (score > bestScore) {
          bestScore = score;
          bestMask = mask;
        }
      }
      
      if (!bestMask) {
        console.warn('Could not find a suitable mask, falling back to traditional detection');
        return this.detectSlidePuzzleGrid(image);
      }
      
      // Step 7: Extract the puzzle contour from the mask
      const contour = this.extractContourFromMask(bestMask);
      if (!contour || contour.length < 4) {
        console.warn('Could not extract a valid contour from the mask, falling back to traditional detection');
        return this.detectSlidePuzzleGrid(image);
      }
      
      // Step 8: Simplify the contour to a quadrilateral (4 points)
      const simplifiedContour = this.approximateToQuadrilateral(contour);
      if (!simplifiedContour || simplifiedContour.length !== 4) {
        console.warn('Could not simplify contour to a quadrilateral, falling back to traditional detection');
        return this.detectSlidePuzzleGrid(image);
      }
      
      console.log('Successfully extracted puzzle quadrilateral from mask');
      
      // Step 9: Extract just the puzzle area using the quadrilateral
      const puzzleCanvas = this.extractPuzzleFromContour(canvas, simplifiedContour);
      
      // Create a puzzle image from the canvas
      const puzzleImage = new Image();
      await new Promise<void>((resolve, reject) => {
        puzzleImage.onload = () => resolve();
        puzzleImage.onerror = () => reject(new Error('Failed to create image from puzzle canvas'));
        puzzleImage.src = puzzleCanvas.toDataURL('image/png');
      });
      
      // Step 10: Use our slide puzzle grid detector on the extracted region
      console.log('Detecting grid in extracted puzzle region...');
      const gridResult = await this.detectSlidePuzzleGrid(puzzleImage);
      
      if (!gridResult) {
        console.warn('Could not detect grid within the puzzle');
        
        // If we can't find a grid, just use a regular grid with the desired size (4x4 for slide puzzles)
        const rows = 4;
        const cols = 4;
        const defaultGridCorners: number[][] = [];
        
        for (let r = 0; r <= rows; r++) {
          for (let c = 0; c <= cols; c++) {
            defaultGridCorners.push([
              c * puzzleCanvas.width / cols,
              r * puzzleCanvas.height / rows
            ]);
          }
        }
        
        // Map the grid to original image coordinates
        const mappedCorners = this.mapGridToOriginalImage(
          defaultGridCorners,
          simplifiedContour,
          offsetX, offsetY, scale
        );
        
        return {
          corners: mappedCorners,
          size: { rows, cols }
        };
      }
      
      // Step 11: Map the grid points back to the original image coordinates
      const mappedCorners = this.mapGridToOriginalImage(
        gridResult.corners,
        simplifiedContour,
        offsetX, offsetY, scale
      );
      
      return {
        corners: mappedCorners,
        size: gridResult.size
      };
      
    } catch (error) {
      console.error('Error detecting puzzle with SAM:', error);
      console.log('Falling back to traditional detection after SAM error');
      return this.detectSlidePuzzleGrid(image);
    }
  }
  
  /**
   * Calculate the area of a mask
   */
  private calculateMaskArea(mask: HTMLCanvasElement): number {
    const ctx = mask.getContext('2d');
    if (!ctx) return 0;
    
    const imageData = ctx.getImageData(0, 0, mask.width, mask.height);
    const data = imageData.data;
    let count = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      // Count white pixels (value > 128)
      if (data[i] > 128) {
        count++;
      }
    }
    
    return count;
  }
  
  /**
   * Calculate how far the mask's center of mass is from the image center
   * Returns a value from 0 (centered) to 1 (at the edge)
   */
  private calculateMaskCenterDistance(mask: HTMLCanvasElement, width: number, height: number): number {
    try {
      const ctx = mask.getContext('2d');
      if (!ctx) return 1;
      
      const imageData = ctx.getImageData(0, 0, mask.width, mask.height);
      const data = imageData.data;
      
      let totalX = 0;
      let totalY = 0;
      let pixelCount = 0;
      
      for (let y = 0; y < mask.height; y++) {
        for (let x = 0; x < mask.width; x++) {
          const idx = (y * mask.width + x) * 4;
          // Check if pixel is part of the mask
          if (data[idx] > 128 || data[idx+1] > 128 || data[idx+2] > 128) {
            totalX += x;
            totalY += y;
            pixelCount++;
          }
        }
      }
      
      if (pixelCount === 0) return 1;
      
      const centerX = totalX / pixelCount;
      const centerY = totalY / pixelCount;
      
      // Calculate normalized distance from center (0-1)
      const dx = (centerX - width/2) / (width/2);
      const dy = (centerY - height/2) / (height/2);
      
      return Math.sqrt(dx*dx + dy*dy);
    } catch (error) {
      console.error('Error calculating mask center distance:', error);
      return 1;
    }
  }
  
  /**
   * Extract a contour from a mask canvas
   */
  private extractContourFromMask(mask: HTMLCanvasElement): number[][] {
    try {
      // Find contours in the mask
      const contours = this.findContours(mask);
      if (!contours || contours.length === 0) {
        return [];
      }
      
      // Find the largest contour by area
      let largestContour: number[][] = [];
      let largestArea = 0;
      
      for (const contour of contours) {
        const area = this.calculateContourArea(contour);
        if (area > largestArea) {
          largestArea = area;
          largestContour = contour;
        }
      }
      
      return largestContour;
    } catch (error) {
      console.error('Error extracting contour from mask:', error);
      return [];
    }
  }
  
  /**
   * Approximate a contour to a quadrilateral (4 points)
   */
  private approximateToQuadrilateral(contour: number[][]): number[][] | null {
    try {
      if (contour.length < 4) return null;
      
      // Implementation of a simple Douglas-Peucker algorithm to simplify contour
      const approxDP = (points: number[][], epsilon: number): number[][] => {
        if (points.length <= 2) return points;
        
        // Find the point with the maximum distance from the line segment
        let maxDistance = 0;
        let maxIndex = 0;
        
        const lineStart = points[0];
        const lineEnd = points[points.length - 1];
        
        for (let i = 1; i < points.length - 1; i++) {
          const distance = this.pointToLineDistanceCalculator(points[i], lineStart, lineEnd);
          if (distance > maxDistance) {
            maxDistance = distance;
            maxIndex = i;
          }
        }
        
        // If max distance is greater than epsilon, recursively simplify
        if (maxDistance > epsilon) {
          const leftPoints = approxDP(points.slice(0, maxIndex + 1), epsilon);
          const rightPoints = approxDP(points.slice(maxIndex), epsilon);
          
          // Combine the results and remove duplicate point
          return [...leftPoints.slice(0, -1), ...rightPoints];
        } else {
          // No need to simplify
          return [lineStart, lineEnd];
        }
      };
      
      // Start with a generous epsilon and gradually decrease until we get 4 points
      let epsilon = 50;
      let simplified = approxDP([...contour], epsilon);
      
      while (simplified.length !== 4 && epsilon > 1) {
        epsilon *= 0.8;
        simplified = approxDP([...contour], epsilon);
      }
      
      // If we couldn't get exactly 4 points, find the 4 points that form the largest area
      if (simplified.length !== 4) {
        console.log(`Could not simplify to exactly 4 points, got ${simplified.length} points`);
        
        // If we have more than 4 points, find the 4 corners that form the largest area
        if (simplified.length > 4) {
          // Simple approach: find extreme points (top-left, top-right, bottom-right, bottom-left)
          const sortByX = [...simplified].sort((a, b) => a[0] - b[0]);
          const leftPoints = sortByX.slice(0, sortByX.length / 2);
          const rightPoints = sortByX.slice(sortByX.length / 2);
          
          const sortLeftByY = [...leftPoints].sort((a, b) => a[1] - b[1]);
          const sortRightByY = [...rightPoints].sort((a, b) => a[1] - b[1]);
          
          const topLeft = sortLeftByY[0];
          const bottomLeft = sortLeftByY[sortLeftByY.length - 1];
          const topRight = sortRightByY[0];
          const bottomRight = sortRightByY[sortRightByY.length - 1];
          
          return [topLeft, topRight, bottomRight, bottomLeft];
        } else {
          return null; // Could not simplify to 4 points
        }
      }
      
      return simplified;
    } catch (error) {
      console.error('Error approximating contour to quadrilateral:', error);
      return null;
    }
  }
  
  /**
   * Calculate distance from a point to a line segment
   */
  private pointToLineDistanceCalculator(point: number[], lineStart: number[], lineEnd: number[]): number {
    const [x0, y0] = point;
    const [x1, y1] = lineStart;
    const [x2, y2] = lineEnd;

    const numerator = Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
    const denominator = Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));

    return numerator / denominator;
  }
  
  /**
   * Extract the puzzle area from the canvas using the contour
   */
  private extractPuzzleFromContour(canvas: HTMLCanvasElement, contour: number[][]): HTMLCanvasElement {
    try {
      // Sort contour points to ensure they are in clockwise order
      // Starting from top-left, going clockwise to top-right, bottom-right, bottom-left
      const sortedContour = [...contour];
      sortedContour.sort((a, b) => {
        const aSum = a[0] + a[1]; // x + y sum
        const bSum = b[0] + b[1];
        return aSum - bSum;
      });
      
      // Top-left has smallest sum of x+y
      const topLeft = sortedContour[0];
      
      // Remove top-left and sort rest by polar angle
      const rest = sortedContour.slice(1);
      rest.sort((a, b) => {
        const angleA = Math.atan2(a[1] - topLeft[1], a[0] - topLeft[0]);
        const angleB = Math.atan2(b[1] - topLeft[1], b[0] - topLeft[0]);
        return angleA - angleB;
      });
      
      // Reordered contour
      const orderedContour = [topLeft, ...rest];
      
      // Create a new canvas for the transformed puzzle
      const result = document.createElement('canvas');
      result.width = 400;
      result.height = 400;
      const resultCtx = result.getContext('2d');
      if (!resultCtx) throw new Error('Failed to get result canvas context');
      
      // Calculate the perspective transform
      const srcPoints = [
        { x: orderedContour[0][0], y: orderedContour[0][1] }, // Top-left
        { x: orderedContour[1][0], y: orderedContour[1][1] }, // Top-right
        { x: orderedContour[2][0], y: orderedContour[2][1] }, // Bottom-right
        { x: orderedContour[3][0], y: orderedContour[3][1] }  // Bottom-left
      ];
      
      const dstPoints = [
        { x: 0, y: 0 },                 // Top-left
        { x: result.width, y: 0 },      // Top-right
        { x: result.width, y: result.height }, // Bottom-right
        { x: 0, y: result.height }      // Bottom-left
      ];
      
      // Apply perspective transform using canvas transform
      resultCtx.setTransform(1, 0, 0, 1, 0, 0);
      resultCtx.clearRect(0, 0, result.width, result.height);
      
      // We need to implement a simplified perspective transform since full perspective
      // is not directly available in canvas. This is a simplified approach.
      const transformFunc = this.getPerspectiveTransform(srcPoints, dstPoints);
      
      // Create temporary canvas for pixel manipulation
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) throw new Error('Failed to get temp canvas context');
      
      tempCtx.drawImage(canvas, 0, 0);
      const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const resultImageData = resultCtx.createImageData(result.width, result.height);
      
      // For each pixel in the destination, find the corresponding pixel in the source
      for (let y = 0; y < result.height; y++) {
        for (let x = 0; x < result.width; x++) {
          // Find the source pixel coordinates using the perspective transform
          const srcCoord = transformFunc.applyInverse(x, y);
          if (srcCoord.x >= 0 && srcCoord.x < canvas.width &&
              srcCoord.y >= 0 && srcCoord.y < canvas.height) {
            // Bilinear interpolation would be better here, but using nearest neighbor for simplicity
            const sx = Math.round(srcCoord.x);
            const sy = Math.round(srcCoord.y);
            
            const srcIdx = (sy * tempCanvas.width + sx) * 4;
            const dstIdx = (y * result.width + x) * 4;
            
            resultImageData.data[dstIdx] = tempImageData.data[srcIdx];
            resultImageData.data[dstIdx + 1] = tempImageData.data[srcIdx + 1];
            resultImageData.data[dstIdx + 2] = tempImageData.data[srcIdx + 2];
            resultImageData.data[dstIdx + 3] = tempImageData.data[srcIdx + 3];
          }
        }
      }
      
      resultCtx.putImageData(resultImageData, 0, 0);
      return result;
    } catch (error) {
      console.error('Error extracting puzzle from contour:', error);
      
      // Return a default square crop of the original canvas
      const result = document.createElement('canvas');
      result.width = 400;
      result.height = 400;
      const resultCtx = result.getContext('2d');
      if (resultCtx) {
        // Draw from original canvas, taking a centered square crop
        const size = Math.min(canvas.width, canvas.height);
        const offsetX = (canvas.width - size) / 2;
        const offsetY = (canvas.height - size) / 2;
        resultCtx.drawImage(canvas, offsetX, offsetY, size, size, 0, 0, 400, 400);
      }
      return result;
    }
  }
  
  // Generate a perspective transform function
  private getPerspectiveTransform(src: { x: number, y: number }[], dst: { x: number, y: number }[]): { 
    apply: (x: number, y: number) => { x: number, y: number },
    applyInverse: (x: number, y: number) => { x: number, y: number }
  } {
    // Simple implementation for transformation
    // Note: This is a simplified version and not a true perspective transform
    return {
      // Map destination to source (inverse transform)
      applyInverse: (x: number, y: number) => {
        // Normalize coordinates to [0,1]
        const nx = x / 400;
        const ny = y / 400;
        
        // Bilinear interpolation
        const srcX = (1 - nx) * (1 - ny) * src[0].x + // Top-left
                      nx * (1 - ny) * src[1].x +      // Top-right
                      nx * ny * src[2].x +            // Bottom-right
                      (1 - nx) * ny * src[3].x;       // Bottom-left
                      
        const srcY = (1 - nx) * (1 - ny) * src[0].y + // Top-left
                      nx * (1 - ny) * src[1].y +      // Top-right
                      nx * ny * src[2].y +            // Bottom-right
                      (1 - nx) * ny * src[3].y;       // Bottom-left
                      
        return { x: srcX, y: srcY };
      },
      
      // Map source to destination (forward transform)
      apply: (x: number, y: number) => {
        // This is a placeholder - we don't actually need this
        // direction for our current implementation
        return { x: 0, y: 0 };
      }
    };
  }
  
  // Map grid corners back to original image coordinates
  private mapGridToOriginalImage(
    gridCorners: number[][], 
    puzzleContour: number[][],
    offsetX: number, 
    offsetY: number, 
    scale: number
  ): number[][] {
    // Find the bounds of the contour
    const minX = Math.min(...puzzleContour.map(p => p[0]));
    const minY = Math.min(...puzzleContour.map(p => p[1]));
    const maxX = Math.max(...puzzleContour.map(p => p[0]));
    const maxY = Math.max(...puzzleContour.map(p => p[1]));
    
    const contourWidth = maxX - minX;
    const contourHeight = maxY - minY;
    
    // Map each grid corner from puzzle coordinates to original image coordinates
    return gridCorners.map(corner => {
      const [x, y] = corner;
      
      // Map from the normalized grid (0-400) to the contour
      const contourX = minX + (x / 400) * contourWidth;
      const contourY = minY + (y / 400) * contourHeight;
      
      // Map from contour to original image
      const originalX = (contourX - offsetX) / scale;
      const originalY = (contourY - offsetY) / scale;
      
      return [originalX, originalY];
    });
  }
} 