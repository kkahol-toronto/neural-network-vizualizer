/**
 * NN-Model Visualizer - Neural Network Engine
 * Implements forward propagation and backpropagation with step-by-step tracking
 */

class NeuralNetwork {
    constructor(config) {
        this.inputSize = config.inputSize || 3;
        this.hiddenLayers = config.hiddenLayers || 2;
        this.neuronsPerLayer = config.neuronsPerLayer || 4;
        this.activationFunction = config.activationFunction || 'sigmoid';
        this.learningRate = config.learningRate || 0.5;
        
        // Network structure
        this.layers = [];
        this.weights = [];
        this.biases = [];
        
        // Forward pass storage
        this.activations = [];
        this.preActivations = []; // z values before activation
        
        // Backward pass storage
        this.deltas = [];
        this.weightGradients = [];
        this.biasGradients = [];
        this.previousWeights = []; // For visualizing changes
        
        // Step tracking for animation
        this.steps = [];
        this.currentStepIndex = -1;
        
        this.initializeNetwork();
    }
    
    initializeNetwork() {
        // Define layer sizes
        this.layers = [this.inputSize];
        for (let i = 0; i < this.hiddenLayers; i++) {
            this.layers.push(this.neuronsPerLayer);
        }
        this.layers.push(1); // Output layer (binary classification)
        
        // Initialize weights and biases with Xavier initialization
        this.weights = [];
        this.biases = [];
        
        for (let i = 0; i < this.layers.length - 1; i++) {
            const inputSize = this.layers[i];
            const outputSize = this.layers[i + 1];
            
            // Xavier initialization
            const limit = Math.sqrt(6 / (inputSize + outputSize));
            
            const layerWeights = [];
            for (let j = 0; j < outputSize; j++) {
                const neuronWeights = [];
                for (let k = 0; k < inputSize; k++) {
                    neuronWeights.push((Math.random() * 2 - 1) * limit);
                }
                layerWeights.push(neuronWeights);
            }
            this.weights.push(layerWeights);
            
            const layerBiases = [];
            for (let j = 0; j < outputSize; j++) {
                layerBiases.push((Math.random() * 2 - 1) * 0.1);
            }
            this.biases.push(layerBiases);
        }
        
        // Store copy for tracking changes
        this.previousWeights = JSON.parse(JSON.stringify(this.weights));
    }
    
    // Activation functions
    activate(x, derivative = false) {
        switch (this.activationFunction) {
            case 'sigmoid':
                if (derivative) {
                    const sig = this.sigmoid(x);
                    return sig * (1 - sig);
                }
                return this.sigmoid(x);
            case 'relu':
                if (derivative) {
                    return x > 0 ? 1 : 0;
                }
                return Math.max(0, x);
            case 'tanh':
                if (derivative) {
                    const t = Math.tanh(x);
                    return 1 - t * t;
                }
                return Math.tanh(x);
            default:
                return this.sigmoid(x);
        }
    }
    
    sigmoid(x) {
        // Clamp to prevent overflow
        x = Math.max(-500, Math.min(500, x));
        return 1 / (1 + Math.exp(-x));
    }
    
    // Forward propagation with step tracking
    forward(input) {
        this.steps = [];
        this.activations = [input.slice()];
        this.preActivations = [null]; // No pre-activation for input layer
        
        let currentActivation = input.slice();
        
        // Add initial step
        this.steps.push({
            type: 'input',
            phase: 'forward',
            layerIndex: 0,
            description: `Input layer receives values: [${input.map(v => v.toFixed(3)).join(', ')}]`,
            highlightNeurons: this.activations[0].map((_, i) => ({ layer: 0, neuron: i })),
            activations: JSON.parse(JSON.stringify(this.activations))
        });
        
        for (let l = 0; l < this.weights.length; l++) {
            const layerWeights = this.weights[l];
            const layerBiases = this.biases[l];
            const newActivation = [];
            const layerPreActivations = [];
            
            for (let j = 0; j < layerWeights.length; j++) {
                // Calculate weighted sum
                let z = layerBiases[j];
                for (let i = 0; i < currentActivation.length; i++) {
                    z += currentActivation[i] * layerWeights[j][i];
                }
                layerPreActivations.push(z);
                
                // Apply activation function
                const a = l === this.weights.length - 1 
                    ? this.sigmoid(z) // Always sigmoid for output layer
                    : this.activate(z);
                newActivation.push(a);
                
                // Add step for each neuron computation
                this.steps.push({
                    type: 'forward_neuron',
                    phase: 'forward',
                    layerIndex: l + 1,
                    neuronIndex: j,
                    description: `Layer ${l + 1}, Neuron ${j + 1}: z = Σ(w·a) + b = ${z.toFixed(4)}, a = ${this.activationFunction}(z) = ${a.toFixed(4)}`,
                    weightedSum: z,
                    activation: a,
                    inputWeights: layerWeights[j].slice(),
                    inputActivations: currentActivation.slice(),
                    bias: layerBiases[j],
                    highlightConnections: currentActivation.map((_, i) => ({
                        fromLayer: l,
                        fromNeuron: i,
                        toLayer: l + 1,
                        toNeuron: j
                    })),
                    highlightNeurons: [{ layer: l + 1, neuron: j }]
                });
            }
            
            this.preActivations.push(layerPreActivations);
            currentActivation = newActivation;
            this.activations.push(newActivation.slice());
            
            // Add layer completion step
            this.steps.push({
                type: 'forward_layer_complete',
                phase: 'forward',
                layerIndex: l + 1,
                description: `Layer ${l + 1} complete: [${newActivation.map(v => v.toFixed(4)).join(', ')}]`,
                activations: JSON.parse(JSON.stringify(this.activations)),
                highlightNeurons: newActivation.map((_, i) => ({ layer: l + 1, neuron: i }))
            });
        }
        
        return currentActivation[0]; // Return output
    }
    
    // Backward propagation with step tracking
    backward(target) {
        const output = this.activations[this.activations.length - 1][0];
        const error = output - target;
        const loss = 0.5 * error * error; // MSE loss
        
        // Store previous weights for visualization
        this.previousWeights = JSON.parse(JSON.stringify(this.weights));
        
        this.deltas = [];
        this.weightGradients = [];
        this.biasGradients = [];
        
        // Add loss calculation step
        this.steps.push({
            type: 'loss',
            phase: 'backward',
            description: `Output: ${output.toFixed(4)}, Target: ${target}, Error: ${error.toFixed(4)}, Loss (MSE): ${loss.toFixed(6)}`,
            output,
            target,
            error,
            loss
        });
        
        // Calculate output layer delta
        // For sigmoid output: δ = (a - y) * σ'(z)
        const outputZ = this.preActivations[this.preActivations.length - 1][0];
        const sigmoidDerivative = output * (1 - output);
        const outputDelta = error * sigmoidDerivative;
        
        this.steps.push({
            type: 'backward_delta',
            phase: 'backward',
            layerIndex: this.layers.length - 1,
            description: `Output layer δ = error × σ'(z) = ${error.toFixed(4)} × ${sigmoidDerivative.toFixed(4)} = ${outputDelta.toFixed(6)}`,
            delta: outputDelta,
            highlightNeurons: [{ layer: this.layers.length - 1, neuron: 0 }]
        });
        
        // Initialize deltas array
        for (let l = 0; l < this.weights.length; l++) {
            this.deltas.push([]);
            this.weightGradients.push([]);
            this.biasGradients.push([]);
        }
        
        this.deltas[this.deltas.length - 1] = [outputDelta];
        
        // Backpropagate deltas through hidden layers
        for (let l = this.weights.length - 2; l >= 0; l--) {
            const layerDeltas = [];
            const nextLayerDeltas = this.deltas[l + 1];
            const nextLayerWeights = this.weights[l + 1];
            
            for (let j = 0; j < this.layers[l + 1]; j++) {
                // Sum of (next layer deltas × weights connecting to this neuron)
                let deltaSum = 0;
                for (let k = 0; k < nextLayerDeltas.length; k++) {
                    deltaSum += nextLayerDeltas[k] * nextLayerWeights[k][j];
                }
                
                // Multiply by activation derivative
                const z = this.preActivations[l + 1][j];
                const activationDerivative = this.activate(z, true);
                const delta = deltaSum * activationDerivative;
                layerDeltas.push(delta);
                
                this.steps.push({
                    type: 'backward_delta',
                    phase: 'backward',
                    layerIndex: l + 1,
                    neuronIndex: j,
                    description: `Layer ${l + 1}, Neuron ${j + 1}: δ = Σ(δₙₑₓₜ × w) × ${this.activationFunction}'(z) = ${delta.toFixed(6)}`,
                    delta,
                    highlightNeurons: [{ layer: l + 1, neuron: j }],
                    highlightConnections: nextLayerDeltas.map((_, k) => ({
                        fromLayer: l + 1,
                        fromNeuron: j,
                        toLayer: l + 2,
                        toNeuron: k
                    }))
                });
            }
            
            this.deltas[l] = layerDeltas;
        }
        
        // Calculate gradients and update weights
        for (let l = 0; l < this.weights.length; l++) {
            const layerGradients = [];
            const layerBiasGradients = [];
            
            for (let j = 0; j < this.weights[l].length; j++) {
                const neuronGradients = [];
                const delta = this.deltas[l][j];
                
                // Bias gradient is just the delta
                layerBiasGradients.push(delta);
                
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    // Weight gradient = delta × input activation
                    const gradient = delta * this.activations[l][i];
                    neuronGradients.push(gradient);
                    
                    // Update weight
                    const oldWeight = this.weights[l][j][i];
                    const weightUpdate = -this.learningRate * gradient;
                    this.weights[l][j][i] += weightUpdate;
                    
                    this.steps.push({
                        type: 'weight_update',
                        phase: 'backward',
                        layerIndex: l,
                        fromNeuron: i,
                        toNeuron: j,
                        description: `w[${l}][${j}][${i}]: ${oldWeight.toFixed(4)} → ${this.weights[l][j][i].toFixed(4)} (Δ = ${weightUpdate.toFixed(6)})`,
                        oldWeight,
                        newWeight: this.weights[l][j][i],
                        gradient,
                        weightUpdate,
                        highlightConnections: [{
                            fromLayer: l,
                            fromNeuron: i,
                            toLayer: l + 1,
                            toNeuron: j
                        }]
                    });
                }
                
                // Update bias
                const oldBias = this.biases[l][j];
                this.biases[l][j] -= this.learningRate * delta;
                
                layerGradients.push(neuronGradients);
            }
            
            this.weightGradients[l] = layerGradients;
            this.biasGradients[l] = layerBiasGradients;
        }
        
        // Add completion step
        this.steps.push({
            type: 'complete',
            phase: 'complete',
            description: `Training step complete. New loss will be calculated on next forward pass.`,
            weights: JSON.parse(JSON.stringify(this.weights)),
            previousWeights: JSON.parse(JSON.stringify(this.previousWeights))
        });
        
        return { loss, output, error };
    }
    
    // Get current step
    getCurrentStep() {
        if (this.currentStepIndex >= 0 && this.currentStepIndex < this.steps.length) {
            return this.steps[this.currentStepIndex];
        }
        return null;
    }
    
    // Navigate steps
    nextStep() {
        if (this.currentStepIndex < this.steps.length - 1) {
            this.currentStepIndex++;
            return this.getCurrentStep();
        }
        return null;
    }
    
    previousStep() {
        if (this.currentStepIndex > 0) {
            this.currentStepIndex--;
            return this.getCurrentStep();
        }
        return null;
    }
    
    resetSteps() {
        this.currentStepIndex = -1;
    }
    
    // Get all network data for visualization
    getNetworkData() {
        return {
            layers: this.layers,
            weights: this.weights,
            biases: this.biases,
            activations: this.activations,
            preActivations: this.preActivations,
            deltas: this.deltas,
            weightGradients: this.weightGradients,
            previousWeights: this.previousWeights
        };
    }
    
    // Train on a single example
    train(input, target) {
        this.resetSteps();
        const output = this.forward(input);
        const result = this.backward(target);
        return { ...result, steps: this.steps };
    }
    
    // Get weight change between previous and current
    getWeightChange(layerIndex, toNeuron, fromNeuron) {
        if (this.previousWeights[layerIndex] && this.weights[layerIndex]) {
            const prev = this.previousWeights[layerIndex][toNeuron][fromNeuron];
            const curr = this.weights[layerIndex][toNeuron][fromNeuron];
            return curr - prev;
        }
        return 0;
    }
}

// Export for use
window.NeuralNetwork = NeuralNetwork;
