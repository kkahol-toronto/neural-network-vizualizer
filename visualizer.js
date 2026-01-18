/**
 * NN-Model Visualizer - Network Visualization
 * Handles SVG rendering of the neural network
 */

class NetworkVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = document.getElementById('network-svg');
        this.network = null;
        
        // Layout configuration
        this.config = {
            neuronRadius: 24,
            layerSpacing: 180,
            neuronSpacing: 70,
            padding: 80,
            connectionWidth: { min: 0.5, max: 4 },
            animationDuration: 600
        };
        
        // Color scheme
        this.colors = {
            neuronPositive: '#00d4ff',
            neuronNegative: '#ff006e',
            neuronNeutral: '#64748b',
            neuronActive: '#10b981',
            connectionPositive: 'rgba(0, 212, 255, 0.6)',
            connectionNegative: 'rgba(255, 0, 110, 0.6)',
            signalForward: '#00d4ff',
            signalBackward: '#ff006e'
        };
        
        // State
        this.neuronElements = [];
        this.connectionElements = [];
        this.signals = [];
        this.tooltip = null;
        this.weightTooltip = null;
        this.equationLabels = [];
        this.computationBox = null;
        
        this.createTooltip();
        this.createWeightTooltip();
        this.createComputationBox();
        this.setupResizeObserver();
    }
    
    createTooltip() {
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tooltip';
        this.tooltip.innerHTML = '<div class="tooltip-title"></div><div class="tooltip-content"></div>';
        document.body.appendChild(this.tooltip);
    }
    
    createWeightTooltip() {
        this.weightTooltip = document.createElement('div');
        this.weightTooltip.className = 'weight-tooltip';
        document.body.appendChild(this.weightTooltip);
    }
    
    showWeightTooltip(x, y, weight, layerIdx, fromNeuron, toNeuron) {
        const change = this.network ? this.network.getWeightChange(layerIdx, toNeuron, fromNeuron) : 0;
        const isPositive = weight >= 0;
        
        let changeHtml = '';
        if (change !== 0) {
            const changeClass = change > 0 ? 'increase' : 'decrease';
            const changeSign = change > 0 ? '+' : '';
            changeHtml = `<div class="wt-change ${changeClass}">Δ ${changeSign}${change.toFixed(6)}</div>`;
        }
        
        this.weightTooltip.className = `weight-tooltip ${isPositive ? 'positive' : 'negative'}`;
        this.weightTooltip.innerHTML = `
            <div class="wt-label">Weight [${layerIdx}][${toNeuron}][${fromNeuron}]</div>
            <div class="wt-value ${isPositive ? 'positive' : 'negative'}">${weight.toFixed(6)}</div>
            ${changeHtml}
        `;
        
        this.weightTooltip.style.left = `${x + 15}px`;
        this.weightTooltip.style.top = `${y - 10}px`;
        
        requestAnimationFrame(() => {
            this.weightTooltip.classList.add('visible');
        });
    }
    
    hideWeightTooltip() {
        this.weightTooltip.classList.remove('visible');
    }
    
    createComputationBox() {
        this.computationBox = document.createElement('div');
        this.computationBox.className = 'computation-box';
        document.body.appendChild(this.computationBox);
    }
    
    // Show equation label on a connection
    showEquationLabel(fromLayer, fromNeuron, toLayer, toNeuron, equation, phase = 'forward') {
        // Get connection position
        if (!this.connectionElements[fromLayer] || 
            !this.connectionElements[fromLayer][toNeuron] || 
            !this.connectionElements[fromLayer][toNeuron][fromNeuron]) {
            return null;
        }
        
        const line = this.connectionElements[fromLayer][toNeuron][fromNeuron];
        const x1 = parseFloat(line.getAttribute('x1'));
        const y1 = parseFloat(line.getAttribute('y1'));
        const x2 = parseFloat(line.getAttribute('x2'));
        const y2 = parseFloat(line.getAttribute('y2'));
        
        // Calculate position at 40% along the line (closer to source)
        const t = 0.4;
        const posX = x1 + (x2 - x1) * t;
        const posY = y1 + (y2 - y1) * t;
        
        // Get SVG position
        const svgRect = this.svg.getBoundingClientRect();
        
        // Create label element
        const label = document.createElement('div');
        label.className = `equation-label ${phase}`;
        label.innerHTML = equation;
        document.body.appendChild(label);
        
        // Get label dimensions after adding to DOM
        const labelRect = label.getBoundingClientRect();
        
        // Position it offset from the line
        // Calculate perpendicular offset based on line angle
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const offsetDistance = 25;
        const offsetX = Math.sin(angle) * offsetDistance;
        const offsetY = -Math.cos(angle) * offsetDistance;
        
        label.style.left = `${svgRect.left + posX + offsetX - labelRect.width / 2}px`;
        label.style.top = `${svgRect.top + posY + offsetY - labelRect.height / 2}px`;
        
        // Show with animation
        requestAnimationFrame(() => {
            label.classList.add('visible');
        });
        
        this.equationLabels.push(label);
        return label;
    }
    
    // Clear all equation labels
    clearEquationLabels() {
        this.equationLabels.forEach(label => {
            label.classList.remove('visible');
            setTimeout(() => label.remove(), 200);
        });
        this.equationLabels = [];
    }
    
    // Show computation box near a neuron
    showComputationBox(layer, neuron, content, phase = 'forward') {
        if (!this.neuronElements[layer] || !this.neuronElements[layer][neuron]) {
            return;
        }
        
        const neuronEl = this.neuronElements[layer][neuron];
        const transform = neuronEl.getAttribute('transform');
        const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
        if (!match) return;
        
        const x = parseFloat(match[1]);
        const y = parseFloat(match[2]);
        
        const svgRect = this.svg.getBoundingClientRect();
        
        this.computationBox.className = `computation-box ${phase}`;
        this.computationBox.innerHTML = content;
        
        // Position to the right of the neuron
        const boxX = svgRect.left + x + this.config.neuronRadius + 20;
        const boxY = svgRect.top + y - 60;
        
        this.computationBox.style.left = `${boxX}px`;
        this.computationBox.style.top = `${boxY}px`;
        
        requestAnimationFrame(() => {
            this.computationBox.classList.add('visible');
        });
    }
    
    // Hide computation box
    hideComputationBox() {
        this.computationBox.classList.remove('visible');
    }
    
    // Generate forward pass equation HTML for a neuron
    generateForwardEquation(step) {
        const weights = step.inputWeights;
        const activations = step.inputActivations;
        const bias = step.bias;
        const z = step.weightedSum;
        const a = step.activation;
        
        let termsHtml = '';
        let valuesHtml = '';
        
        for (let i = 0; i < weights.length; i++) {
            termsHtml += `w${i}·a${i}`;
            valuesHtml += `${weights[i].toFixed(3)}×${activations[i].toFixed(3)}`;
            if (i < weights.length - 1) {
                termsHtml += ' + ';
                valuesHtml += ' + ';
            }
        }
        
        return `
            <div class="comp-title">Forward Pass Computation</div>
            <div class="comp-step">
                <div class="comp-label">Weighted Sum (z)</div>
                <div class="comp-equation">z = Σ(wᵢ × aᵢ) + b</div>
                <div class="comp-values">${valuesHtml} + ${bias.toFixed(3)}</div>
                <div class="comp-values"><strong>z = ${z.toFixed(4)}</strong></div>
            </div>
            <div class="comp-step">
                <div class="comp-label">Activation</div>
                <div class="comp-equation">a = σ(z)</div>
            </div>
            <div class="comp-result">
                <div class="comp-result-label">Output Activation</div>
                <div class="comp-result-value">${a.toFixed(4)}</div>
            </div>
        `;
    }
    
    // Generate backward pass equation HTML for weight update
    generateWeightUpdateEquation(step, learningRate) {
        const oldWeight = step.oldWeight;
        const gradient = step.gradient;
        const newWeight = step.newWeight;
        const update = step.weightUpdate;
        
        return `
            <div class="comp-title">Weight Update</div>
            <div class="comp-step">
                <div class="comp-label">Gradient</div>
                <div class="comp-equation">∂L/∂w = δ × a</div>
                <div class="comp-values">${gradient.toFixed(6)}</div>
            </div>
            <div class="comp-step">
                <div class="comp-label">Update Rule</div>
                <div class="comp-equation">w_new = w_old - η × gradient</div>
                <div class="comp-values">${oldWeight.toFixed(4)} - ${learningRate} × ${gradient.toFixed(4)}</div>
            </div>
            <div class="comp-result">
                <div class="comp-result-label">New Weight</div>
                <div class="comp-result-value">${newWeight.toFixed(4)}</div>
            </div>
        `;
    }
    
    // Generate delta computation equation
    generateDeltaEquation(step, isOutput = false) {
        if (isOutput) {
            return `
                <div class="comp-title">Output Delta Computation</div>
                <div class="comp-step">
                    <div class="comp-label">Error Signal</div>
                    <div class="comp-equation">δ = (output - target) × σ'(z)</div>
                </div>
                <div class="comp-result">
                    <div class="comp-result-label">Delta (δ)</div>
                    <div class="comp-result-value">${step.delta.toFixed(6)}</div>
                </div>
            `;
        } else {
            return `
                <div class="comp-title">Hidden Layer Delta</div>
                <div class="comp-step">
                    <div class="comp-label">Backpropagated Error</div>
                    <div class="comp-equation">δ = Σ(δₙₑₓₜ × w) × σ'(z)</div>
                </div>
                <div class="comp-result">
                    <div class="comp-result-label">Delta (δ)</div>
                    <div class="comp-result-value">${step.delta.toFixed(6)}</div>
                </div>
            `;
        }
    }
    
    // Generate loss computation equation
    generateLossEquation(step) {
        return `
            <div class="comp-title">Loss Calculation</div>
            <div class="comp-step">
                <div class="comp-label">Network Output</div>
                <div class="comp-values">${step.output.toFixed(4)}</div>
            </div>
            <div class="comp-step">
                <div class="comp-label">Target</div>
                <div class="comp-values">${step.target}</div>
            </div>
            <div class="comp-step">
                <div class="comp-label">Error</div>
                <div class="comp-equation">error = output - target</div>
                <div class="comp-values">${step.error.toFixed(4)}</div>
            </div>
            <div class="comp-step">
                <div class="comp-label">Mean Squared Error</div>
                <div class="comp-equation">L = ½ × error²</div>
            </div>
            <div class="comp-result">
                <div class="comp-result-label">Loss</div>
                <div class="comp-result-value">${step.loss.toFixed(6)}</div>
            </div>
        `;
    }
    
    showTooltip(x, y, title, content) {
        const titleEl = this.tooltip.querySelector('.tooltip-title');
        const contentEl = this.tooltip.querySelector('.tooltip-content');
        titleEl.textContent = title;
        contentEl.innerHTML = content;
        
        this.tooltip.style.left = `${x + 15}px`;
        this.tooltip.style.top = `${y - 10}px`;
        this.tooltip.classList.add('visible');
    }
    
    hideTooltip() {
        this.tooltip.classList.remove('visible');
    }
    
    setupResizeObserver() {
        const resizeObserver = new ResizeObserver(() => {
            if (this.network) {
                this.render(this.network);
            }
        });
        resizeObserver.observe(this.container);
    }
    
    // Set the network to visualize
    setNetwork(network) {
        this.network = network;
        this.render(network);
    }
    
    // Main render function
    render(network) {
        if (!network) return;
        
        this.network = network;
        const data = network.getNetworkData();
        
        // Clear SVG
        this.svg.innerHTML = '';
        this.neuronElements = [];
        this.connectionElements = [];
        
        // Create defs for gradients and filters
        this.createDefs();
        
        // Calculate layout
        const layout = this.calculateLayout(data.layers);
        
        // Create groups for layering
        const connectionsGroup = this.createGroup('connections');
        const neuronsGroup = this.createGroup('neurons');
        const labelsGroup = this.createGroup('labels');
        
        // Draw connections first (behind neurons)
        this.drawConnections(data, layout, connectionsGroup);
        
        // Draw neurons
        this.drawNeurons(data, layout, neuronsGroup);
        
        // Draw labels
        this.drawLabels(data, layout, labelsGroup);
    }
    
    createDefs() {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Gradient for positive neurons
        const gradientPos = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
        gradientPos.setAttribute('id', 'neuronGradientPositive');
        gradientPos.innerHTML = `
            <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#00d4ff" stop-opacity="0.1"/>
        `;
        defs.appendChild(gradientPos);
        
        // Gradient for negative neurons
        const gradientNeg = document.createElementNS('http://www.w3.org/2000/svg', 'radialGradient');
        gradientNeg.setAttribute('id', 'neuronGradientNegative');
        gradientNeg.innerHTML = `
            <stop offset="0%" stop-color="#ff006e" stop-opacity="0.3"/>
            <stop offset="100%" stop-color="#ff006e" stop-opacity="0.1"/>
        `;
        defs.appendChild(gradientNeg);
        
        // Glow filter
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'glow');
        filter.innerHTML = `
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        `;
        defs.appendChild(filter);
        
        // Signal glow filter
        const signalFilter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        signalFilter.setAttribute('id', 'signalGlow');
        signalFilter.innerHTML = `
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>
        `;
        defs.appendChild(signalFilter);
        
        this.svg.appendChild(defs);
    }
    
    createGroup(id) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('id', id);
        this.svg.appendChild(group);
        return group;
    }
    
    calculateLayout(layers) {
        const svgRect = this.svg.getBoundingClientRect();
        const width = svgRect.width || 800;
        const height = svgRect.height || 500;
        
        const numLayers = layers.length;
        const maxNeurons = Math.max(...layers);
        
        // Calculate spacing
        const usableWidth = width - (this.config.padding * 2);
        const usableHeight = height - (this.config.padding * 2);
        
        const layerSpacing = usableWidth / (numLayers - 1);
        
        const positions = [];
        
        for (let l = 0; l < numLayers; l++) {
            const layerPositions = [];
            const numNeurons = layers[l];
            
            // Center neurons vertically
            const totalHeight = (numNeurons - 1) * this.config.neuronSpacing;
            const startY = (height - totalHeight) / 2;
            
            for (let n = 0; n < numNeurons; n++) {
                layerPositions.push({
                    x: this.config.padding + (l * layerSpacing),
                    y: startY + (n * this.config.neuronSpacing)
                });
            }
            
            positions.push(layerPositions);
        }
        
        return { positions, width, height };
    }
    
    drawConnections(data, layout, group) {
        this.connectionElements = [];
        
        for (let l = 0; l < data.weights.length; l++) {
            const layerConnections = [];
            
            for (let j = 0; j < data.weights[l].length; j++) {
                const neuronConnections = [];
                
                for (let i = 0; i < data.weights[l][j].length; i++) {
                    const weight = data.weights[l][j][i];
                    const fromPos = layout.positions[l][i];
                    const toPos = layout.positions[l + 1][j];
                    
                    // Create connection line
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', fromPos.x);
                    line.setAttribute('y1', fromPos.y);
                    line.setAttribute('x2', toPos.x);
                    line.setAttribute('y2', toPos.y);
                    line.setAttribute('class', 'connection');
                    
                    // Style based on weight
                    const absWeight = Math.abs(weight);
                    const maxWeight = 2;
                    const normalizedWeight = Math.min(absWeight / maxWeight, 1);
                    const strokeWidth = this.config.connectionWidth.min + 
                        normalizedWeight * (this.config.connectionWidth.max - this.config.connectionWidth.min);
                    
                    line.setAttribute('stroke-width', strokeWidth);
                    
                    if (weight >= 0) {
                        const opacity = 0.2 + normalizedWeight * 0.5;
                        line.setAttribute('stroke', `rgba(0, 212, 255, ${opacity})`);
                    } else {
                        const opacity = 0.2 + normalizedWeight * 0.5;
                        line.setAttribute('stroke', `rgba(255, 0, 110, ${opacity})`);
                    }
                    
                    // Store metadata
                    line.dataset.layer = l;
                    line.dataset.from = i;
                    line.dataset.to = j;
                    line.dataset.weight = weight;
                    
                    // Add hover event for weight tooltip
                    line.addEventListener('mouseenter', (e) => {
                        const currentWeight = parseFloat(line.dataset.weight);
                        this.showWeightTooltip(e.clientX, e.clientY, currentWeight, l, i, j);
                    });
                    line.addEventListener('mousemove', (e) => {
                        this.weightTooltip.style.left = `${e.clientX + 15}px`;
                        this.weightTooltip.style.top = `${e.clientY - 10}px`;
                    });
                    line.addEventListener('mouseleave', () => this.hideWeightTooltip());
                    
                    group.appendChild(line);
                    neuronConnections.push(line);
                }
                
                layerConnections.push(neuronConnections);
            }
            
            this.connectionElements.push(layerConnections);
        }
    }
    
    drawNeurons(data, layout, group) {
        this.neuronElements = [];
        
        for (let l = 0; l < data.layers.length; l++) {
            const layerNeurons = [];
            
            for (let n = 0; n < data.layers[l]; n++) {
                const pos = layout.positions[l][n];
                const neuronGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                neuronGroup.setAttribute('class', 'neuron');
                neuronGroup.setAttribute('transform', `translate(${pos.x}, ${pos.y})`);
                
                // Get activation value if available
                let activation = 0;
                if (data.activations && data.activations[l]) {
                    activation = data.activations[l][n] || 0;
                }
                
                // Outer glow circle
                const glowCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                glowCircle.setAttribute('r', this.config.neuronRadius + 4);
                glowCircle.setAttribute('fill', activation >= 0.5 ? 'url(#neuronGradientPositive)' : 'url(#neuronGradientNegative)');
                glowCircle.setAttribute('opacity', Math.abs(activation - 0.5) * 2 * 0.5 + 0.2);
                neuronGroup.appendChild(glowCircle);
                
                // Main neuron circle
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('r', this.config.neuronRadius);
                circle.setAttribute('class', 'neuron-circle');
                
                // Color based on activation
                const color = this.getActivationColor(activation);
                circle.setAttribute('fill', `rgba(${color.r}, ${color.g}, ${color.b}, 0.2)`);
                circle.setAttribute('stroke', `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`);
                
                neuronGroup.appendChild(circle);
                
                // Value text
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('class', 'neuron-value');
                text.textContent = activation.toFixed(2);
                neuronGroup.appendChild(text);
                
                // Store metadata
                neuronGroup.dataset.layer = l;
                neuronGroup.dataset.neuron = n;
                
                // Add hover event
                neuronGroup.addEventListener('mouseenter', (e) => {
                    let content = `Activation: ${activation.toFixed(6)}`;
                    if (data.preActivations && data.preActivations[l] && data.preActivations[l][n] !== undefined) {
                        content += `<br>Pre-activation (z): ${data.preActivations[l][n].toFixed(6)}`;
                    }
                    if (data.deltas && data.deltas[l - 1] && data.deltas[l - 1][n] !== undefined) {
                        content += `<br>Delta (δ): ${data.deltas[l - 1][n].toFixed(6)}`;
                    }
                    
                    let layerName = l === 0 ? 'Input' : l === data.layers.length - 1 ? 'Output' : `Hidden ${l}`;
                    this.showTooltip(e.clientX, e.clientY, 
                        `${layerName} - Neuron ${n + 1}`,
                        content
                    );
                });
                neuronGroup.addEventListener('mouseleave', () => this.hideTooltip());
                
                group.appendChild(neuronGroup);
                layerNeurons.push(neuronGroup);
            }
            
            this.neuronElements.push(layerNeurons);
        }
    }
    
    drawLabels(data, layout, group) {
        const layerNames = ['Input', ...Array(data.layers.length - 2).fill(0).map((_, i) => `Hidden ${i + 1}`), 'Output'];
        
        for (let l = 0; l < data.layers.length; l++) {
            const firstNeuronPos = layout.positions[l][0];
            
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', firstNeuronPos.x);
            text.setAttribute('y', firstNeuronPos.y - this.config.neuronRadius - 20);
            text.setAttribute('class', 'neuron-label');
            text.textContent = layerNames[l];
            group.appendChild(text);
        }
    }
    
    getActivationColor(value) {
        // Interpolate between negative (magenta) and positive (cyan)
        if (value >= 0.5) {
            const t = (value - 0.5) * 2; // 0 to 1
            return {
                r: Math.round(0 + t * 0),
                g: Math.round(150 + t * 62),
                b: Math.round(180 + t * 75)
            };
        } else {
            const t = value * 2; // 0 to 1
            return {
                r: Math.round(255 - t * 155),
                g: Math.round(0 + t * 100),
                b: Math.round(110 + t * 70)
            };
        }
    }
    
    // Highlight specific neurons
    highlightNeurons(neurons, clear = true) {
        // Clear previous highlights
        if (clear) {
            this.neuronElements.flat().forEach(el => {
                el.classList.remove('neuron-active');
            });
        }
        
        neurons.forEach(({ layer, neuron }) => {
            if (this.neuronElements[layer] && this.neuronElements[layer][neuron]) {
                const el = this.neuronElements[layer][neuron];
                el.classList.add('neuron-active');
            }
        });
    }
    
    // Highlight specific connections
    highlightConnections(connections, phase = 'forward', clear = true) {
        // Clear previous highlights
        if (clear) {
            this.connectionElements.flat().flat().forEach(el => {
                el.classList.remove('connection-active');
                el.style.stroke = '';
                el.style.strokeWidth = '';
            });
        }
        
        connections.forEach(({ fromLayer, fromNeuron, toLayer, toNeuron }) => {
            const layerIdx = fromLayer;
            if (this.connectionElements[layerIdx] && 
                this.connectionElements[layerIdx][toNeuron] && 
                this.connectionElements[layerIdx][toNeuron][fromNeuron]) {
                const el = this.connectionElements[layerIdx][toNeuron][fromNeuron];
                el.classList.add('connection-active');
                el.style.strokeWidth = '3px';
                el.style.stroke = phase === 'forward' ? '#00d4ff' : '#ff006e';
            }
        });
    }
    
    // Animate signal along a connection
    animateSignal(fromLayer, fromNeuron, toLayer, toNeuron, phase = 'forward', duration = 500) {
        if (!this.connectionElements[fromLayer] || 
            !this.connectionElements[fromLayer][toNeuron] || 
            !this.connectionElements[fromLayer][toNeuron][fromNeuron]) {
            return Promise.resolve();
        }
        
        const line = this.connectionElements[fromLayer][toNeuron][fromNeuron];
        const x1 = parseFloat(line.getAttribute('x1'));
        const y1 = parseFloat(line.getAttribute('y1'));
        const x2 = parseFloat(line.getAttribute('x2'));
        const y2 = parseFloat(line.getAttribute('y2'));
        
        // Create signal element
        const signal = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        signal.setAttribute('r', 5);
        signal.setAttribute('class', phase === 'forward' ? 'signal' : 'signal-backward');
        signal.setAttribute('filter', 'url(#signalGlow)');
        
        // Determine direction
        const startX = phase === 'forward' ? x1 : x2;
        const startY = phase === 'forward' ? y1 : y2;
        const endX = phase === 'forward' ? x2 : x1;
        const endY = phase === 'forward' ? y2 : y1;
        
        signal.setAttribute('cx', startX);
        signal.setAttribute('cy', startY);
        
        this.svg.appendChild(signal);
        this.signals.push(signal);
        
        return new Promise(resolve => {
            const startTime = performance.now();
            
            const animate = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                // Ease out cubic
                const eased = 1 - Math.pow(1 - progress, 3);
                
                const cx = startX + (endX - startX) * eased;
                const cy = startY + (endY - startY) * eased;
                
                signal.setAttribute('cx', cx);
                signal.setAttribute('cy', cy);
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    signal.remove();
                    const idx = this.signals.indexOf(signal);
                    if (idx > -1) this.signals.splice(idx, 1);
                    resolve();
                }
            };
            
            requestAnimationFrame(animate);
        });
    }
    
    // Clear all signals
    clearSignals() {
        this.signals.forEach(signal => signal.remove());
        this.signals = [];
    }
    
    // Update neuron values
    updateNeuronValue(layer, neuron, value) {
        if (this.neuronElements[layer] && this.neuronElements[layer][neuron]) {
            const neuronEl = this.neuronElements[layer][neuron];
            const text = neuronEl.querySelector('.neuron-value');
            const circle = neuronEl.querySelector('.neuron-circle');
            const glow = neuronEl.querySelector('circle:first-child');
            
            if (text) text.textContent = value.toFixed(2);
            
            // Update colors
            const color = this.getActivationColor(value);
            if (circle) {
                circle.setAttribute('fill', `rgba(${color.r}, ${color.g}, ${color.b}, 0.2)`);
                circle.setAttribute('stroke', `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`);
            }
            if (glow) {
                glow.setAttribute('fill', value >= 0.5 ? 'url(#neuronGradientPositive)' : 'url(#neuronGradientNegative)');
                glow.setAttribute('opacity', Math.abs(value - 0.5) * 2 * 0.5 + 0.2);
            }
        }
    }
    
    // Update connection weight visualization
    updateConnectionWeight(layer, fromNeuron, toNeuron, weight, change = 0) {
        if (this.connectionElements[layer] && 
            this.connectionElements[layer][toNeuron] && 
            this.connectionElements[layer][toNeuron][fromNeuron]) {
            const line = this.connectionElements[layer][toNeuron][fromNeuron];
            
            const absWeight = Math.abs(weight);
            const maxWeight = 2;
            const normalizedWeight = Math.min(absWeight / maxWeight, 1);
            const strokeWidth = this.config.connectionWidth.min + 
                normalizedWeight * (this.config.connectionWidth.max - this.config.connectionWidth.min);
            
            line.setAttribute('stroke-width', strokeWidth);
            
            // Highlight changes
            if (change !== 0) {
                line.style.stroke = change > 0 ? '#10b981' : '#f97316';
                setTimeout(() => {
                    if (weight >= 0) {
                        const opacity = 0.2 + normalizedWeight * 0.5;
                        line.style.stroke = `rgba(0, 212, 255, ${opacity})`;
                    } else {
                        const opacity = 0.2 + normalizedWeight * 0.5;
                        line.style.stroke = `rgba(255, 0, 110, ${opacity})`;
                    }
                }, 500);
            } else {
                if (weight >= 0) {
                    const opacity = 0.2 + normalizedWeight * 0.5;
                    line.setAttribute('stroke', `rgba(0, 212, 255, ${opacity})`);
                } else {
                    const opacity = 0.2 + normalizedWeight * 0.5;
                    line.setAttribute('stroke', `rgba(255, 0, 110, ${opacity})`);
                }
            }
            
            line.dataset.weight = weight;
        }
    }
    
    // Clear all highlights
    clearHighlights() {
        this.neuronElements.flat().forEach(el => {
            el.classList.remove('neuron-active');
        });
        this.connectionElements.flat().flat().forEach(el => {
            el.classList.remove('connection-active');
            el.style.strokeWidth = '';
        });
        this.clearSignals();
        this.clearEquationLabels();
        this.hideComputationBox();
    }
}


// Export for use
window.NetworkVisualizer = NetworkVisualizer;
