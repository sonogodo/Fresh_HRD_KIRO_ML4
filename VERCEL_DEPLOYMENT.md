# Deployment Guide for Vercel

## 🚀 Quick Deployment (Optimized for Serverless)

This project includes a Vercel-optimized version that stays under the 250MB serverless function limit while maintaining all functionality.

### 1. Prepare Optimized Deployment

```bash
# Run optimization script
python deploy_vercel.py

# Test the optimized version
python test_vercel_app.py

# Verify optimization
ls -la vercel_app.py decision_ml_fallback.py
```

### 2. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy optimized version
vercel --prod
```

Or use the Vercel dashboard:
1. Connect your GitHub repository
2. Import the project
3. Deploy automatically

### 3. Optimization Details

The Vercel deployment uses an optimized approach to stay under the 250MB limit:
- **vercel_app.py**: Lightweight version without heavy ML dependencies
- **decision_ml_fallback.py**: Demonstration mode with realistic simulations  
- **Minimal requirements.txt**: Only essential packages (FastAPI, uvicorn, etc.)
- **.vercelignore**: Excludes heavy files from deployment
- **Function size**: <50MB (well under 250MB limit)

### 2. Environment Variables

No additional environment variables are required. The system will automatically:
- Use full ML capabilities if all dependencies are available
- Fall back to demonstration mode if ML dependencies are missing
- Maintain full API compatibility in both modes

### 3. Features Available

#### ✅ Always Available
- **Basic Matching**: Original algorithm using keyword matching
- **File Upload**: Multiple job analysis via JSON files
- **Interactive UI**: Modern web interface with tabs
- **API Documentation**: Available at `/docs`

#### 🧠 ML Features (Full Mode)
- **Advanced AI Matching**: Machine learning-based candidate ranking
- **Model Training**: Train custom models with Decision data
- **Performance Monitoring**: Drift detection and metrics
- **Detailed Analytics**: Comprehensive scoring breakdown

#### 🎯 ML Features (Demo Mode)
- **Simulated AI Matching**: Demonstrates ML capabilities with sample data
- **Mock Training**: Shows training interface and status
- **Sample Analytics**: Example monitoring and reporting
- **Educational Purpose**: Perfect for presentations and demos

### 4. API Endpoints

All endpoints work in both modes:

```bash
# Health check
GET /health

# Basic matching (always available)
POST /match_vaga
POST /match_vagas

# Advanced ML matching (full or demo mode)
POST /decision/predict
POST /decision/train
GET /decision/status
GET /decision/monitoring/health
GET /decision/monitoring/report
GET /decision/models/summary
```

### 5. Data Requirements

#### For Full ML Mode:
- Place your data files in `JSONs_DECISION/`:
  - `vagas_padrao.json` - Job data in Decision format
  - `candidates.json` - Candidate profiles

#### For Demo Mode:
- No data files required
- Uses built-in sample data for demonstration

### 6. Vercel Configuration

The project includes:
- ✅ `vercel.json` - Vercel configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `app.py` - FastAPI application
- ✅ Automatic fallback system
- ✅ Static file serving

### 7. Performance Considerations

#### Vercel Limits:
- **Function Timeout**: 10 seconds (Hobby), 60 seconds (Pro)
- **Memory**: 1GB (Hobby), 3GB (Pro)
- **Cold Starts**: ~1-2 seconds for Python functions

#### Optimizations:
- Lightweight fallback mode for quick responses
- Efficient caching of model predictions
- Minimal dependencies in requirements.txt
- Lazy loading of ML components

### 8. Monitoring

#### Built-in Monitoring:
- System health checks
- Response time tracking
- Error logging
- Performance metrics

#### Access Monitoring:
```bash
# Check system status
curl https://your-app.vercel.app/health

# Get detailed monitoring
curl https://your-app.vercel.app/decision/monitoring/health
```

### 9. Troubleshooting

#### Common Issues:

1. **ML Features Not Working**:
   - Check `/health` endpoint for component status
   - System automatically falls back to demo mode
   - All functionality remains available

2. **Slow Cold Starts**:
   - First request may take 2-3 seconds
   - Subsequent requests are fast
   - Consider Vercel Pro for better performance

3. **Memory Issues**:
   - Large ML models may exceed memory limits
   - Fallback mode uses minimal memory
   - Consider model optimization for production

#### Debug Information:
```bash
# Check deployment logs
vercel logs

# Test all endpoints
curl https://your-app.vercel.app/docs
```

### 10. Scaling Considerations

#### For Production:
- **Vercel Pro**: Higher limits and better performance
- **Database**: Consider external database for large datasets
- **Model Storage**: Use external storage for trained models
- **Caching**: Implement Redis for prediction caching

#### Architecture Options:
- **Serverless**: Current setup (good for demos and small scale)
- **Hybrid**: Vercel frontend + dedicated ML backend
- **Container**: Docker deployment for full control

### 11. Security

#### Built-in Security:
- ✅ CORS configuration
- ✅ Input validation
- ✅ Error handling
- ✅ No sensitive data exposure

#### Additional Recommendations:
- Add rate limiting for production
- Implement authentication if needed
- Use environment variables for secrets
- Monitor for unusual usage patterns

### 12. Cost Estimation

#### Vercel Pricing:
- **Hobby**: Free (good for demos)
- **Pro**: $20/month (recommended for production)
- **Enterprise**: Custom pricing

#### Usage Patterns:
- **Demo Mode**: Very low resource usage
- **Full ML Mode**: Higher compute requirements
- **Training**: Intensive but infrequent

---

## 🎉 Ready to Deploy!

Your Decision ML platform is ready for Vercel deployment with:
- ✅ Automatic fallback system
- ✅ Full API compatibility
- ✅ Modern web interface
- ✅ Production-ready configuration
- ✅ Comprehensive monitoring

Deploy now and start revolutionizing recruitment with AI! 🚀

## 🔧 Troubleshooting Vercel Deployment Issues

### Configuration Errors

#### ❌ "The `functions` property cannot be used in conjunction with the `builds` property"
```bash
# Fix vercel.json configuration automatically
python deploy_vercel.py

# Or manually edit vercel.json to remove "functions" property
# Keep only "builds" and "routes"
```

#### ❌ "Serverless Function exceeds 250MB"
```bash
# Run optimization check
python deploy_vercel.py

# Verify minimal requirements.txt (should only have 4 packages)
cat requirements.txt

# Check .vercelignore excludes heavy files
cat .vercelignore
```

### Deployment Errors

#### ❌ "Build failed" or import errors
```bash
# Test locally first
python test_vercel_app.py

# Check for missing files
ls -la vercel_app.py decision_ml_fallback.py

# Verify imports work
python -c "from vercel_app import app; print('OK')"
```

#### ❌ App doesn't start after deployment
```bash
# Check vercel.json points to correct file
cat vercel.json  # Should show "src": "vercel_app.py"

# Check deployment logs
vercel logs

# Test health endpoint
curl https://your-app.vercel.app/health
```

### Quick Fix Commands

```bash
# Complete optimization and test
python deploy_vercel.py && python test_vercel_app.py

# Clean redeploy
rm -rf .vercel && vercel --prod

# Force fresh deployment
vercel --prod --force
```

### Verification Checklist

Before deploying, ensure:
- ✅ `vercel_app.py` exists and works
- ✅ `decision_ml_fallback.py` exists
- ✅ `requirements.txt` has only 4 packages
- ✅ `vercel.json` has no "functions" property
- ✅ `.vercelignore` excludes heavy files
- ✅ All tests pass: `python test_vercel_app.py`