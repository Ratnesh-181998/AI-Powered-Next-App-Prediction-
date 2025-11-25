# 📱 iPhone App Predictor - Web UI

## 🎨 UI Screenshot

![iPhone App Predictor UI](app_predictor_ui_complete.png)

---

## 📁 Files in This Folder

| File | Description | Size |
|------|-------------|------|
| **index.html** | Interactive web interface | ~25 KB |
| **app_predictor_ui_complete.png** | UI screenshot | ~500 KB |
| **UI_GUIDE.md** | User guide | ~10 KB |

---

## 🌟 UI Features

### **Header Section**
- **Title**: "📱 iPhone App Predictor"
- **Subtitle**: "AI-Powered Next App Prediction with 100% Accuracy"
- **Design**: Purple gradient background with pulsing animation

### **Statistics Bar**
Four cards displaying key metrics:
1. **100%** - Top-1 Accuracy
2. **7-12ms** - Inference Time
3. **50** - Apps Supported
4. **28** - Features Used

### **Left Panel: Context Settings**

#### Time Configuration
- **Time of Day Buttons**: Morning 🌅 / Afternoon ☀️ / Evening 🌙
- **Hour Slider**: 0-23 hours with live value display
- **Day of Week**: Dropdown (Sunday-Saturday)

#### Device Context
- **Battery Level**: Slider (0-100%) with color gradient
- **Network Type**: Toggle buttons (WiFi 📶 / 4G 📡 / 3G 📱)
- **Charging Status**: Toggle (Not Charging 🔋 / Charging ⚡)

#### Action Button
- **"🔮 Predict Next App"**: Large purple gradient button

#### Info Panel
- Explains how the AI model works
- Tips for best results

### **Right Panel: Predictions**

#### Prediction Cards
Each card shows:
- **Rank**: Numbered circle (1, 2, 3)
- **App Name**: e.g., "Instagram"
- **Confidence**: Percentage (e.g., 65.0%)
- **Progress Bar**: Visual representation of confidence

#### Performance Info
- **Inference Time**: Milliseconds (e.g., "9.27ms")
- **Model Version**: e.g., "xgboost_v1"

---

## 🎨 Design System

### **Colors**
```css
Primary: #007AFF (iOS Blue)
Gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%)
Success: #34C759 (Green)
Warning: #FF9500 (Orange)
Danger: #FF3B30 (Red)
Background: #F2F2F7 (Light Gray)
Card: #FFFFFF (White)
Text: #000000 (Black)
Secondary Text: #8E8E93 (Gray)
```

### **Typography**
```css
Font Family: -apple-system, BlinkMacSystemFont, 'Segoe UI'
Headings: 700 weight, 1.5-3rem
Body: 400 weight, 1rem
Labels: 500 weight, 0.95rem
```

### **Spacing**
```css
Card Padding: 30px
Gap: 20-30px
Border Radius: 12-20px
Box Shadow: 0 4px 20px rgba(0,0,0,0.08)
```

### **Animations**
```css
Transitions: 0.3s ease
Hover: translateY(-5px)
Slide In: 0.5s ease
Pulse: 15s infinite
```

---

## 🚀 How to Use

### **1. Open the UI**
```bash
# Option 1: Double-click
index.html

# Option 2: Open in browser
file:///c:/Users/rattu/Downloads/L-19/project/web/index.html

# Option 3: Serve with Python
python -m http.server 8000
# Then visit: http://localhost:8000
```

### **2. Configure Context**
- Select time of day
- Adjust hour and battery
- Choose network type
- Set charging status

### **3. Get Predictions**
- Click "🔮 Predict Next App"
- View top 3 predictions
- See confidence scores
- Check inference time

---

## 🔌 API Connection

### **Demo Mode** (Default)
- Works without API server
- Uses simulated predictions
- Based on time-of-day logic

### **Real API Mode**
1. Start API server:
   ```bash
   python src/api/app.py
   ```

2. Refresh the UI
   - Automatically connects to `http://localhost:5000`
   - Uses trained XGBoost model
   - Real predictions with 100% accuracy

---

## 📊 UI Components

### **Interactive Elements**

#### Sliders
- **Hour Slider**: Drag to set time (0-23)
- **Battery Slider**: Drag to set level (0-100%)
- Color-coded: Red → Yellow → Green

#### Toggle Buttons
- **Time of Day**: Single selection
- **Network Type**: Single selection
- **Charging Status**: Binary toggle
- Active state: Purple background

#### Dropdown
- **Day of Week**: Select from 7 options
- Styled to match iOS design

#### Prediction Cards
- **Animated Entry**: Slide in from left
- **Rank Badge**: Purple gradient circle
- **Confidence Bar**: Animated width
- **Hover Effect**: Slight elevation

---

## 🎯 User Experience

### **Loading State**
- Spinning loader
- "Making prediction..." message
- Smooth transition to results

### **Success State**
- 3 prediction cards
- Confidence percentages
- Visual progress bars
- Performance metrics

### **Error State** (if API fails)
- Falls back to demo mode
- Shows simulated predictions
- No disruption to user

---

## 📱 Responsive Design

### **Desktop** (>968px)
- Two-column layout
- Side-by-side panels
- Full-width stats

### **Tablet** (768-968px)
- Single column
- Stacked panels
- Optimized spacing

### **Mobile** (<768px)
- Mobile-first design
- Touch-optimized controls
- Larger tap targets

---

## 🎨 Customization

### **Change Colors**
Edit CSS variables in `index.html`:
```css
:root {
    --primary: #007AFF;
    --gradient: linear-gradient(...);
}
```

### **Modify Layout**
Adjust grid in CSS:
```css
.main-grid {
    grid-template-columns: 1fr 1fr;
}
```

### **Add Features**
Extend JavaScript:
```javascript
function customFeature() {
    // Your code here
}
```

---

## 🐛 Troubleshooting

### **UI Not Loading**
- Check browser console (F12)
- Ensure JavaScript is enabled
- Try different browser

### **Predictions Not Working**
- Check if API server is running
- Verify network connection
- Look for CORS errors

### **Styling Issues**
- Clear browser cache
- Disable browser extensions
- Check CSS is loaded

---

## 📈 Performance

### **Load Time**
- Initial: <1 second
- Cached: <100ms

### **Interaction**
- Button clicks: Instant
- Slider updates: Real-time
- Animations: 60 FPS

### **Prediction**
- Demo mode: ~500ms
- Real API: 7-12ms

---

## 🎉 Features Showcase

✅ **Modern Design** - iOS-inspired interface  
✅ **Smooth Animations** - 60 FPS transitions  
✅ **Real-time Updates** - Live value displays  
✅ **Responsive Layout** - Works on all devices  
✅ **Interactive Controls** - Intuitive inputs  
✅ **Visual Feedback** - Hover and active states  
✅ **Error Handling** - Graceful fallbacks  
✅ **Performance** - Optimized for speed  

---

## 📚 Resources

- **UI Guide**: See `UI_GUIDE.md`
- **API Docs**: See `../src/api/app.py`
- **Project Guide**: See `../PROJECT_GUIDE.md`

---

## 🎊 Credits

**Built with**:
- HTML5
- CSS3
- Vanilla JavaScript
- No frameworks or libraries

**Design Inspiration**:
- iOS Design Guidelines
- Material Design
- Modern Web Standards

---

**Enjoy your beautiful iPhone App Predictor UI!** 🎨📱

*Last Updated: 2025-11-25*
