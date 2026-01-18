/**
 * NN-Model Visualizer - Main Application
 * Handles user interactions, playback, and step-by-step animation
 */

class App {
    constructor() {
        // State
        this.network = null;
        this.visualizer = null;
        this.isPlaying = false;
        this.animationSpeed = 1;
        this.currentInput = [];
        this.currentTarget = 0;
        this.trainingResult = null;
        
        // Animation state
        this.currentStepIndex = -1;
        this.animationFrame = null;
        this.stepTimeout = null;
        
        // Epoch training state
        this.isEpochTraining = false;
        this.currentEpoch = 0;
        this.currentSampleInEpoch = 0;
        this.samplesPerEpoch = 5;
        this.numEpochs = 5;
        this.epochLosses = [];
        this.epochTimeout = null;
        
        // Initialize
        this.init();
    }
    
    init() {
        // Create visualizer
        this.visualizer = new NetworkVisualizer('network-container');
        
        // Setup event listeners
        this.setupControls();
        this.setupSliders();
        this.setupPlayback();
        this.setupTabs();
        this.setupHelpModal();
        this.setupEpochControls();
        
        // Build initial network
        this.buildNetwork();
    }
    
    setupEpochControls() {
        // Samples per epoch slider
        const samplesSlider = document.getElementById('samples-per-epoch');
        const samplesDisplay = document.getElementById('samples-per-epoch-display');
        samplesSlider.addEventListener('input', (e) => {
            this.samplesPerEpoch = parseInt(e.target.value);
            samplesDisplay.textContent = this.samplesPerEpoch;
            this.updateEpochDisplay();
        });
        
        // Number of epochs slider
        const epochsSlider = document.getElementById('num-epochs');
        const epochsDisplay = document.getElementById('num-epochs-display');
        epochsSlider.addEventListener('input', (e) => {
            this.numEpochs = parseInt(e.target.value);
            epochsDisplay.textContent = this.numEpochs;
            this.updateEpochDisplay();
        });
        
        // Run epoch button
        document.getElementById('run-epoch').addEventListener('click', () => {
            this.startEpochTraining();
        });
        
        // Stop epoch button
        document.getElementById('stop-epoch').addEventListener('click', () => {
            this.stopEpochTraining();
        });
    }
    
    updateEpochDisplay() {
        document.getElementById('current-epoch').textContent = `${this.currentEpoch} / ${this.numEpochs}`;
        document.getElementById('current-sample').textContent = `${this.currentSampleInEpoch} / ${this.samplesPerEpoch}`;
        
        // Calculate progress
        const totalSamples = this.numEpochs * this.samplesPerEpoch;
        const completedSamples = (this.currentEpoch * this.samplesPerEpoch) + this.currentSampleInEpoch;
        const progress = totalSamples > 0 ? (completedSamples / totalSamples) * 100 : 0;
        document.getElementById('epoch-bar').style.width = `${progress}%`;
        
        // Update average loss
        if (this.epochLosses.length > 0) {
            const avgLoss = this.epochLosses.reduce((a, b) => a + b, 0) / this.epochLosses.length;
            document.getElementById('avg-loss').textContent = avgLoss.toFixed(6);
        }
    }
    
    startEpochTraining() {
        if (!this.network) {
            this.buildNetwork();
        }
        
        this.isEpochTraining = true;
        this.currentEpoch = 0;
        this.currentSampleInEpoch = 0;
        this.epochLosses = [];
        
        // Update UI
        document.getElementById('run-epoch').style.display = 'none';
        document.getElementById('stop-epoch').style.display = 'flex';
        
        this.updateEpochDisplay();
        this.runNextEpochSample();
    }
    
    stopEpochTraining() {
        this.isEpochTraining = false;
        
        if (this.epochTimeout) {
            clearTimeout(this.epochTimeout);
            this.epochTimeout = null;
        }
        
        // Update UI
        document.getElementById('run-epoch').style.display = 'flex';
        document.getElementById('stop-epoch').style.display = 'none';
    }
    
    runNextEpochSample() {
        if (!this.isEpochTraining) return;
        
        // Check if we've completed all epochs
        if (this.currentEpoch >= this.numEpochs) {
            this.stopEpochTraining();
            this.updatePhaseIndicator('Training Complete!');
            return;
        }
        
        // Generate new sample
        this.generateNewSample();
        
        // Run training
        this.trainingResult = this.network.train(this.currentInput, this.currentTarget);
        this.epochLosses.push(this.trainingResult.loss);
        
        // Update display
        document.getElementById('current-output').textContent = this.trainingResult.output.toFixed(4);
        document.getElementById('current-loss').textContent = this.trainingResult.loss.toFixed(6);
        
        // Update visualization
        this.visualizer.render(this.network);
        this.updateInfoPanel();
        
        // Update epoch progress
        this.currentSampleInEpoch++;
        
        // Check if epoch is complete
        if (this.currentSampleInEpoch >= this.samplesPerEpoch) {
            this.currentEpoch++;
            this.currentSampleInEpoch = 0;
            
            // Calculate epoch average loss
            const epochStartIdx = (this.currentEpoch - 1) * this.samplesPerEpoch;
            const epochLosses = this.epochLosses.slice(epochStartIdx);
            const epochAvgLoss = epochLosses.reduce((a, b) => a + b, 0) / epochLosses.length;
            
            this.updatePhaseIndicator(`Epoch ${this.currentEpoch} Complete - Avg Loss: ${epochAvgLoss.toFixed(4)}`);
        } else {
            this.updatePhaseIndicator(`Epoch ${this.currentEpoch + 1} - Sample ${this.currentSampleInEpoch}/${this.samplesPerEpoch}`);
        }
        
        this.updateEpochDisplay();
        
        // Schedule next sample
        const delay = 300 / this.animationSpeed;
        this.epochTimeout = setTimeout(() => {
            this.runNextEpochSample();
        }, delay);
    }
    
    setupHelpModal() {
        const helpBtn = document.getElementById('help-btn');
        const helpModal = document.getElementById('help-modal');
        const helpClose = document.getElementById('help-close');
        const helpGotIt = document.getElementById('help-got-it');
        
        const openModal = () => {
            helpModal.classList.add('visible');
            document.body.style.overflow = 'hidden';
        };
        
        const closeModal = () => {
            helpModal.classList.remove('visible');
            document.body.style.overflow = '';
        };
        
        helpBtn.addEventListener('click', openModal);
        helpClose.addEventListener('click', closeModal);
        helpGotIt.addEventListener('click', closeModal);
        
        // Close on overlay click
        helpModal.addEventListener('click', (e) => {
            if (e.target === helpModal) {
                closeModal();
            }
        });
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && helpModal.classList.contains('visible')) {
                closeModal();
            }
        });
    }
    
    setupControls() {
        // Build network button
        document.getElementById('build-network').addEventListener('click', () => {
            this.buildNetwork();
        });
        
        // Randomize weights button
        document.getElementById('randomize-weights').addEventListener('click', () => {
            if (this.network) {
                this.network.initializeNetwork();
                this.visualizer.render(this.network);
                this.reset();
                this.updateInfoPanel();
            }
        });
        
        // New sample button
        document.getElementById('new-sample').addEventListener('click', () => {
            this.generateNewSample();
        });
    }
    
    setupSliders() {
        // Input count slider
        const inputSlider = document.getElementById('input-count');
        const inputDisplay = document.getElementById('input-count-display');
        inputSlider.addEventListener('input', () => {
            inputDisplay.textContent = inputSlider.value;
        });
        
        // Hidden layers slider
        const hiddenSlider = document.getElementById('hidden-layers');
        const hiddenDisplay = document.getElementById('hidden-layers-display');
        hiddenSlider.addEventListener('input', () => {
            hiddenDisplay.textContent = hiddenSlider.value;
        });
        
        // Neurons per layer slider
        const neuronsSlider = document.getElementById('neurons-per-layer');
        const neuronsDisplay = document.getElementById('neurons-per-layer-display');
        neuronsSlider.addEventListener('input', () => {
            neuronsDisplay.textContent = neuronsSlider.value;
        });
        
        // Learning rate slider
        const lrSlider = document.getElementById('learning-rate');
        const lrDisplay = document.getElementById('learning-rate-display');
        lrSlider.addEventListener('input', () => {
            lrDisplay.textContent = parseFloat(lrSlider.value).toFixed(2);
            if (this.network) {
                this.network.learningRate = parseFloat(lrSlider.value);
            }
        });
        
        // Animation speed slider
        const speedSlider = document.getElementById('animation-speed');
        const speedDisplay = document.getElementById('speed-display');
        speedSlider.addEventListener('input', () => {
            this.animationSpeed = parseFloat(speedSlider.value);
            speedDisplay.textContent = `${this.animationSpeed}x`;
        });
    }
    
    setupPlayback() {
        // Play/Pause button
        document.getElementById('play-pause').addEventListener('click', () => {
            this.togglePlayback();
        });
        
        // Step forward
        document.getElementById('step-forward').addEventListener('click', () => {
            this.stepForward();
        });
        
        // Step back
        document.getElementById('step-back').addEventListener('click', () => {
            this.stepBack();
        });
        
        // Reset button
        document.getElementById('reset').addEventListener('click', () => {
            this.reset();
        });
    }
    
    setupTabs() {
        const tabs = document.querySelectorAll('.info-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.updateInfoPanel(tab.dataset.tab);
            });
        });
    }
    
    buildNetwork() {
        const config = {
            inputSize: parseInt(document.getElementById('input-count').value),
            hiddenLayers: parseInt(document.getElementById('hidden-layers').value),
            neuronsPerLayer: parseInt(document.getElementById('neurons-per-layer').value),
            activationFunction: document.getElementById('activation-function').value,
            learningRate: parseFloat(document.getElementById('learning-rate').value)
        };
        
        this.network = new NeuralNetwork(config);
        this.visualizer.setNetwork(this.network);
        
        // Generate initial sample
        this.generateNewSample();
        
        // Reset state
        this.reset();
        
        this.updateInfoPanel();
    }
    
    generateNewSample() {
        if (!this.network) return;
        
        // Generate random input values between 0 and 1
        this.currentInput = [];
        for (let i = 0; i < this.network.inputSize; i++) {
            this.currentInput.push(Math.random());
        }
        
        // Generate target (simple logic: 1 if average > 0.5, else 0)
        const avg = this.currentInput.reduce((a, b) => a + b, 0) / this.currentInput.length;
        this.currentTarget = avg > 0.5 ? 1 : 0;
        
        // Update display
        document.getElementById('current-input').textContent = 
            `[${this.currentInput.map(v => v.toFixed(3)).join(', ')}]`;
        document.getElementById('current-target').textContent = this.currentTarget;
        document.getElementById('current-output').textContent = '--';
        document.getElementById('current-loss').textContent = '--';
        
        // Reset and train
        this.reset();
        this.runTraining();
    }
    
    runTraining() {
        if (!this.network) return;
        
        this.trainingResult = this.network.train(this.currentInput, this.currentTarget);
        
        // Update display
        document.getElementById('current-output').textContent = this.trainingResult.output.toFixed(4);
        document.getElementById('current-loss').textContent = this.trainingResult.loss.toFixed(6);
        
        // Update visualization
        this.visualizer.render(this.network);
        this.updateInfoPanel();
        
        // Reset step index
        this.currentStepIndex = -1;
        this.updatePhaseIndicator('Ready');
    }
    
    togglePlayback() {
        if (this.isPlaying) {
            this.pausePlayback();
        } else {
            this.startPlayback();
        }
    }
    
    startPlayback() {
        if (!this.trainingResult || !this.trainingResult.steps.length) {
            this.runTraining();
        }
        
        this.isPlaying = true;
        this.updatePlayButton();
        
        // If at the end, restart
        if (this.currentStepIndex >= this.trainingResult.steps.length - 1) {
            this.currentStepIndex = -1;
        }
        
        this.playNextStep();
    }
    
    pausePlayback() {
        this.isPlaying = false;
        this.updatePlayButton();
        
        if (this.stepTimeout) {
            clearTimeout(this.stepTimeout);
            this.stepTimeout = null;
        }
    }
    
    playNextStep() {
        if (!this.isPlaying) return;
        
        if (this.currentStepIndex >= this.trainingResult.steps.length - 1) {
            this.pausePlayback();
            return;
        }
        
        this.currentStepIndex++;
        const step = this.trainingResult.steps[this.currentStepIndex];
        
        this.executeStep(step).then(() => {
            // Calculate delay based on animation speed
            const baseDelay = 800;
            const delay = baseDelay / this.animationSpeed;
            
            this.stepTimeout = setTimeout(() => {
                this.playNextStep();
            }, delay);
        });
    }
    
    stepForward() {
        if (!this.trainingResult || !this.trainingResult.steps.length) {
            this.runTraining();
            return;
        }
        
        if (this.currentStepIndex >= this.trainingResult.steps.length - 1) {
            return;
        }
        
        this.currentStepIndex++;
        const step = this.trainingResult.steps[this.currentStepIndex];
        this.executeStep(step);
    }
    
    stepBack() {
        if (!this.trainingResult || this.currentStepIndex <= 0) {
            return;
        }
        
        this.currentStepIndex--;
        
        // Re-render from the beginning up to this step
        this.visualizer.clearHighlights();
        const step = this.trainingResult.steps[this.currentStepIndex];
        this.executeStep(step);
    }
    
    async executeStep(step) {
        // Update phase indicator
        this.updatePhaseIndicator(this.getPhaseText(step));
        
        // Clear previous highlights
        this.visualizer.clearHighlights();
        
        // Execute based on step type
        switch (step.type) {
            case 'input':
                await this.animateInputStep(step);
                break;
                
            case 'forward_neuron':
                await this.animateForwardNeuronStep(step);
                break;
                
            case 'forward_layer_complete':
                await this.animateLayerCompleteStep(step);
                break;
                
            case 'loss':
                await this.animateLossStep(step);
                break;
                
            case 'backward_delta':
                await this.animateBackwardDeltaStep(step);
                break;
                
            case 'weight_update':
                await this.animateWeightUpdateStep(step);
                break;
                
            case 'complete':
                await this.animateCompleteStep(step);
                break;
        }
        
        // Update info panel
        this.updateInfoPanel();
        this.updateStepDescription(step.description);
    }
    
    async animateInputStep(step) {
        if (step.highlightNeurons) {
            this.visualizer.highlightNeurons(step.highlightNeurons);
        }
        
        // Update input neuron values
        if (step.activations && step.activations[0]) {
            for (let i = 0; i < step.activations[0].length; i++) {
                this.visualizer.updateNeuronValue(0, i, step.activations[0][i]);
            }
        }
        
        // Add pulsing animation to input neurons
        await this.delay(300 / this.animationSpeed);
    }
    
    async animateForwardNeuronStep(step) {
        // Clear previous equation labels
        this.visualizer.clearEquationLabels();
        
        // Highlight incoming connections
        if (step.highlightConnections) {
            this.visualizer.highlightConnections(step.highlightConnections, 'forward');
            
            // Animate signals along connections ONE BY ONE with equation labels
            if (step.inputWeights && step.inputActivations) {
                for (let i = 0; i < step.highlightConnections.length; i++) {
                    const conn = step.highlightConnections[i];
                    const w = step.inputWeights[i];
                    const a = step.inputActivations[i];
                    const product = w * a;
                    
                    // Clear previous and show current equation
                    this.visualizer.clearEquationLabels();
                    
                    const equationHtml = `
                        <div class="eq-title">Weight × Input [${i + 1}/${step.highlightConnections.length}]</div>
                        <div class="eq-formula">w<sub>${i}</sub> × a<sub>${i}</sub></div>
                        <div class="eq-values">${w.toFixed(3)} × ${a.toFixed(3)}</div>
                        <div class="eq-result">= ${product.toFixed(4)}</div>
                    `;
                    
                    this.visualizer.showEquationLabel(
                        conn.fromLayer, 
                        conn.fromNeuron, 
                        conn.toLayer, 
                        conn.toNeuron,
                        equationHtml,
                        'forward'
                    );
                    
                    // Animate signal for this connection
                    await this.visualizer.animateSignal(
                        conn.fromLayer, 
                        conn.fromNeuron, 
                        conn.toLayer, 
                        conn.toNeuron, 
                        'forward',
                        250 / this.animationSpeed
                    );
                }
            } else {
                // Fallback: animate all at once
                const signalPromises = step.highlightConnections.map(conn => 
                    this.visualizer.animateSignal(
                        conn.fromLayer, 
                        conn.fromNeuron, 
                        conn.toLayer, 
                        conn.toNeuron, 
                        'forward',
                        400 / this.animationSpeed
                    )
                );
                await Promise.all(signalPromises);
            }
        }
        
        // Clear connection labels and show computation box with full equation
        this.visualizer.clearEquationLabels();
        if (step.inputWeights) {
            const equationContent = this.visualizer.generateForwardEquation(step);
            this.visualizer.showComputationBox(step.layerIndex, step.neuronIndex, equationContent, 'forward');
        }
        
        // Highlight the neuron
        if (step.highlightNeurons) {
            this.visualizer.highlightNeurons(step.highlightNeurons);
        }
        
        // Update neuron value
        this.visualizer.updateNeuronValue(step.layerIndex, step.neuronIndex, step.activation);
        
        await this.delay(300 / this.animationSpeed);
    }
    
    async animateLayerCompleteStep(step) {
        if (step.highlightNeurons) {
            this.visualizer.highlightNeurons(step.highlightNeurons);
        }
        
        await this.delay(200 / this.animationSpeed);
    }
    
    async animateLossStep(step) {
        // Clear previous labels
        this.visualizer.clearEquationLabels();
        
        // Highlight output neuron
        const outputLayer = this.network.layers.length - 1;
        this.visualizer.highlightNeurons([{ layer: outputLayer, neuron: 0 }]);
        
        // Show loss computation box
        const lossEquation = this.visualizer.generateLossEquation(step);
        this.visualizer.showComputationBox(outputLayer, 0, lossEquation, 'backward');
        
        // Update loss display
        document.getElementById('current-output').textContent = step.output.toFixed(4);
        document.getElementById('current-loss').textContent = step.loss.toFixed(6);
        
        await this.delay(800 / this.animationSpeed);
    }
    
    async animateBackwardDeltaStep(step) {
        // Clear previous labels
        this.visualizer.clearEquationLabels();
        
        if (step.highlightNeurons) {
            this.visualizer.highlightNeurons(step.highlightNeurons);
        }
        
        if (step.highlightConnections) {
            this.visualizer.highlightConnections(step.highlightConnections, 'backward', false);
        }
        
        // Show delta computation box
        const isOutputLayer = step.layerIndex === this.network.layers.length - 1;
        const deltaEquation = this.visualizer.generateDeltaEquation(step, isOutputLayer);
        
        const neuronIdx = step.neuronIndex !== undefined ? step.neuronIndex : 0;
        this.visualizer.showComputationBox(step.layerIndex, neuronIdx, deltaEquation, 'backward');
        
        await this.delay(500 / this.animationSpeed);
    }
    
    async animateWeightUpdateStep(step) {
        // Clear previous labels
        this.visualizer.clearEquationLabels();
        this.visualizer.hideComputationBox();
        
        if (step.highlightConnections) {
            this.visualizer.highlightConnections(step.highlightConnections, 'backward');
            
            const conn = step.highlightConnections[0];
            if (conn) {
                // Show weight update equation on the connection
                const equationHtml = `
                    <div class="eq-title">Weight Update</div>
                    <div class="eq-formula">w_new = w - η×∂L/∂w</div>
                    <div class="eq-values">${step.oldWeight.toFixed(3)} - ${this.network.learningRate}×${step.gradient.toFixed(4)}</div>
                    <div class="eq-result">= ${step.newWeight.toFixed(4)}</div>
                `;
                
                this.visualizer.showEquationLabel(
                    conn.fromLayer,
                    conn.fromNeuron,
                    conn.toLayer,
                    conn.toNeuron,
                    equationHtml,
                    'backward'
                );
                
                // Animate signal backward
                await this.visualizer.animateSignal(
                    conn.fromLayer,
                    conn.fromNeuron,
                    conn.toLayer,
                    conn.toNeuron,
                    'backward',
                    300 / this.animationSpeed
                );
            }
        }
        
        // Update connection weight visualization
        this.visualizer.updateConnectionWeight(
            step.layerIndex,
            step.fromNeuron,
            step.toNeuron,
            step.newWeight,
            step.weightUpdate
        );
        
        await this.delay(200 / this.animationSpeed);
    }
    
    async animateCompleteStep(step) {
        // Clear all visual elements
        this.visualizer.clearHighlights();
        this.visualizer.clearEquationLabels();
        this.visualizer.hideComputationBox();
        
        // Re-render with final weights
        this.visualizer.render(this.network);
        
        await this.delay(300 / this.animationSpeed);
    }
    
    reset() {
        this.pausePlayback();
        this.currentStepIndex = -1;
        this.visualizer.clearHighlights();
        this.visualizer.clearEquationLabels();
        this.visualizer.hideComputationBox();
        this.updatePhaseIndicator('Ready');
        this.updateStepDescription('Click Play to start step-by-step visualization, or use Step buttons');
        
        // Re-render fresh
        if (this.network) {
            this.visualizer.render(this.network);
        }
    }
    
    updatePlayButton() {
        const playIcon = document.getElementById('play-icon');
        const pauseIcon = document.getElementById('pause-icon');
        
        if (this.isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    }
    
    updatePhaseIndicator(text) {
        const phaseDot = document.getElementById('phase-forward');
        const phaseText = document.getElementById('current-phase');
        
        phaseText.textContent = text;
        
        // Update dot style based on phase
        phaseDot.classList.remove('forward', 'backward');
        
        if (text.toLowerCase().includes('forward') || text.toLowerCase().includes('input')) {
            phaseDot.classList.add('forward');
        } else if (text.toLowerCase().includes('backward') || text.toLowerCase().includes('loss') || text.toLowerCase().includes('delta') || text.toLowerCase().includes('weight')) {
            phaseDot.classList.add('backward');
        }
    }
    
    updateStepDescription(description) {
        // Could add a dedicated element for this
        console.log('Step:', description);
    }
    
    getPhaseText(step) {
        switch (step.type) {
            case 'input': return 'Forward Pass - Input';
            case 'forward_neuron': return `Forward Pass - Layer ${step.layerIndex}`;
            case 'forward_layer_complete': return `Forward Pass - Layer ${step.layerIndex} Complete`;
            case 'loss': return 'Loss Calculation';
            case 'backward_delta': return `Backward Pass - Computing δ`;
            case 'weight_update': return 'Backward Pass - Updating Weights';
            case 'complete': return 'Training Step Complete';
            default: return step.phase || 'Processing';
        }
    }
    
    updateInfoPanel(activeTab) {
        if (!this.network) return;
        
        const tab = activeTab || document.querySelector('.info-tab.active')?.dataset.tab || 'weights';
        const infoContent = document.getElementById('info-content');
        const data = this.network.getNetworkData();
        
        let html = '';
        
        switch (tab) {
            case 'weights':
                html = this.renderWeightsInfo(data);
                break;
            case 'gradients':
                html = this.renderGradientsInfo(data);
                break;
            case 'activations':
                html = this.renderActivationsInfo(data);
                break;
        }
        
        infoContent.innerHTML = html || '<div class="info-placeholder">No data available</div>';
        
        // Setup click handlers for weight items
        if (tab === 'weights') {
            this.setupWeightClickHandlers();
        }
    }
    
    renderWeightsInfo(data) {
        let html = '<div class="weight-grid">';
        
        for (let l = 0; l < data.weights.length; l++) {
            for (let j = 0; j < data.weights[l].length; j++) {
                for (let i = 0; i < data.weights[l][j].length; i++) {
                    const weight = data.weights[l][j][i];
                    const change = this.network.getWeightChange(l, j, i);
                    const changeClass = change > 0 ? 'increase' : change < 0 ? 'decrease' : '';
                    const valueClass = weight >= 0 ? 'positive' : 'negative';
                    
                    html += `
                        <div class="weight-item" data-layer="${l}" data-from="${i}" data-to="${j}" data-weight="${weight}">
                            <span class="weight-label">w[${l}][${j}][${i}]</span>
                            <span class="weight-value ${valueClass}">${weight.toFixed(4)}</span>
                            ${change !== 0 ? `<span class="weight-change ${changeClass}">${change >= 0 ? '+' : ''}${change.toFixed(6)}</span>` : ''}
                        </div>
                    `;
                }
            }
        }
        
        html += '</div>';
        return html;
    }
    
    setupWeightClickHandlers() {
        const weightItems = document.querySelectorAll('.weight-item[data-layer]');
        weightItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const layer = parseInt(item.dataset.layer);
                const from = parseInt(item.dataset.from);
                const to = parseInt(item.dataset.to);
                const weight = parseFloat(item.dataset.weight);
                
                // Remove previous selection
                document.querySelectorAll('.weight-item.selected').forEach(el => {
                    el.classList.remove('selected', 'negative');
                });
                
                // Add selection to clicked item
                item.classList.add('selected');
                if (weight < 0) {
                    item.classList.add('negative');
                }
                
                // Highlight the corresponding connection
                this.highlightWeightConnection(layer, from, to);
            });
        });
    }
    
    highlightWeightConnection(layer, fromNeuron, toNeuron) {
        // Clear previous highlights
        this.visualizer.clearHighlights();
        
        // Highlight the specific connection
        this.visualizer.highlightConnections([{
            fromLayer: layer,
            fromNeuron: fromNeuron,
            toLayer: layer + 1,
            toNeuron: toNeuron
        }], 'forward', true);
        
        // Also highlight the connected neurons
        this.visualizer.highlightNeurons([
            { layer: layer, neuron: fromNeuron },
            { layer: layer + 1, neuron: toNeuron }
        ], false);
    }
    
    renderGradientsInfo(data) {
        if (!data.weightGradients || data.weightGradients.length === 0 || 
            data.weightGradients.every(layer => layer.length === 0)) {
            return '<div class="info-placeholder">Run a training step to see gradients</div>';
        }
        
        let html = '<div class="weight-grid">';
        
        for (let l = 0; l < data.weightGradients.length; l++) {
            for (let j = 0; j < data.weightGradients[l].length; j++) {
                for (let i = 0; i < data.weightGradients[l][j].length; i++) {
                    const gradient = data.weightGradients[l][j][i];
                    const valueClass = gradient >= 0 ? 'positive' : 'negative';
                    
                    html += `
                        <div class="weight-item">
                            <span class="weight-label">∂L/∂w[${l}][${j}][${i}]</span>
                            <span class="weight-value ${valueClass}">${gradient.toFixed(6)}</span>
                        </div>
                    `;
                }
            }
        }
        
        html += '</div>';
        return html;
    }
    
    renderActivationsInfo(data) {
        if (!data.activations || data.activations.length === 0) {
            return '<div class="info-placeholder">Run forward pass to see activations</div>';
        }
        
        let html = '<div class="weight-grid">';
        
        const layerNames = ['Input', ...Array(data.layers.length - 2).fill(0).map((_, i) => `Hidden ${i + 1}`), 'Output'];
        
        for (let l = 0; l < data.activations.length; l++) {
            for (let n = 0; n < data.activations[l].length; n++) {
                const activation = data.activations[l][n];
                const valueClass = activation >= 0.5 ? 'positive' : 'negative';
                
                html += `
                    <div class="weight-item">
                        <span class="weight-label">${layerNames[l]}[${n}]</span>
                        <span class="weight-value ${valueClass}">${activation.toFixed(4)}</span>
                    </div>
                `;
            }
        }
        
        html += '</div>';
        return html;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
