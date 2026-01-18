# 🧠 NN-Model Visualizer

An interactive neural network visualization tool to understand how **forward propagation** and **backpropagation** work. Watch signals flow through the network, see equations compute in real-time, and observe how weights change during training.

## 🌐 [**Try it Live →**](https://ashy-water-0abed3810.6.azurestaticapps.net)

![NN-Model Visualizer](https://img.shields.io/badge/Neural_Network-Visualizer-00d4ff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![JavaScript](https://img.shields.io/badge/JavaScript-Vanilla-f7df1e?style=for-the-badge&logo=javascript&logoColor=black)
![Azure](https://img.shields.io/badge/Deployed_on-Azure-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)

## ✨ Features

### Network Configuration
- **Configurable Architecture**: Set input neurons (1-6), hidden layers (1-4), and neurons per layer (2-8)
- **Activation Functions**: Choose between Sigmoid, ReLU, or Tanh
- **Learning Rate Control**: Adjust how fast the network learns (0.01-1.0)

### Interactive Visualization
- **Real-time Network Display**: SVG-based visualization with animated signals
- **Color-coded Weights**: Cyan for positive, magenta for negative weights
- **Weight Thickness**: Line thickness indicates weight magnitude
- **Hover Tooltips**: Hover over any connection to see weight values and changes
- **Click to Highlight**: Click weight cells in the matrix to highlight corresponding edges

### Step-by-Step Animation
- **Forward Pass**: Watch input signals propagate through each layer
- **Equation Display**: See the computation (w × a) appear on each connection
- **Computation Box**: View full summation and activation calculations
- **Backward Pass**: Visualize gradient flow and weight updates
- **Playback Controls**: Play, pause, step forward/back, and adjust speed

### Epoch Training
- **Samples per Epoch**: Configure 1-10 training samples per epoch
- **Multiple Epochs**: Run up to 10 epochs of training
- **Progress Tracking**: Watch epoch/sample counters and average loss
- **Loss Visualization**: Observe how the network learns over time

### Glass-Morphic UI
- Beautiful dark theme with translucent panels
- Gradient accents and subtle animations
- Responsive layout with smooth transitions

## 🚀 Getting Started

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/kkahol-toronto/neural-network-vizualizer.git
   cd neural-network-vizualizer
   ```

2. Start a local server:
   ```bash
   python3 -m http.server 8080
   ```

3. Open your browser to `http://localhost:8080`

### Usage

1. **Configure Network**: Use the sliders to set up your neural network architecture
2. **Build Network**: Click "Build Network" to create the network with random weights
3. **Run Animation**: 
   - Click **Play** to watch the full forward and backward pass
   - Use **Step Forward/Back** for manual control
4. **Epoch Training**: 
   - Set samples per epoch and number of epochs
   - Click **Run Training** to watch the network learn
5. **Explore**:
   - Hover over connections to see weight values
   - Click weight cells in the matrix to highlight edges
   - Switch between Weights/Gradients/Activations tabs

## 📁 Project Structure

```
neural-network-vizualizer/
├── index.html          # Main HTML structure
├── styles.css          # Glass-morphic styling
├── neural-network.js   # Core NN engine (forward/backward prop)
├── visualizer.js       # SVG visualization and animations
├── app.js              # Application logic and UI handling
└── README.md           # This file
```

## 🎓 Educational Value

This visualizer helps understand:

- **Forward Propagation**: How inputs are transformed layer by layer
- **Activation Functions**: The role of non-linearity (Sigmoid, ReLU, Tanh)
- **Loss Calculation**: How prediction error is measured
- **Backpropagation**: How gradients flow backward through the network
- **Weight Updates**: How learning rate affects weight changes
- **Epochs & Training**: How repeated training improves the network

## 🛠️ Technical Details

- **Pure JavaScript**: No frameworks or dependencies
- **SVG Rendering**: Scalable vector graphics for crisp visualization
- **Xavier Initialization**: Proper weight initialization for stable training
- **Binary Classification**: Single output neuron with binary cross-entropy loss

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with ❤️ for anyone learning about neural networks.

---

**Made by [Kanav Kahol](https://github.com/kkahol-toronto)**
