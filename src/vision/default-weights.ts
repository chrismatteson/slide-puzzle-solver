import { type ClassifierWeights } from "./classifier.ts";
import weights from "./weights.json" with { type: "json" };

// Trained by `npm run train` (scripts/train_classifier.py) on fonts/train only.
export const defaultWeights: ClassifierWeights = weights;
