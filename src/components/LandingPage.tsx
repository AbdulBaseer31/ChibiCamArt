import { useState } from 'react'
import { Button } from './ui/button'
import { Camera, Layers, ArrowRight } from 'lucide-react'

interface LandingPageProps {
  onModeSelect: (mode: 'webcam' | 'wireframe') => void;
}

export function LandingPage({ onModeSelect }: LandingPageProps) {
  const [step, setStep] = useState<'welcome' | 'mode-select'>('welcome')

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-purple-900 to-fuchsia-900 flex items-center justify-center p-4 text-white selection:bg-purple-500/30 overflow-hidden relative">
      
      {/* Decorative background effects */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-500/20 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-fuchsia-500/20 rounded-full blur-[120px] mix-blend-screen"></div>
      </div>

      <div className="max-w-4xl w-full flex flex-col items-center justify-center text-center space-y-8 z-10">
        
        {step === 'welcome' ? (
          <div className="animate-in fade-in zoom-in-95 duration-1000 flex flex-col items-center">
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tighter drop-shadow-2xl bg-clip-text text-transparent bg-gradient-to-b from-white via-purple-100 to-purple-300 pb-4">
              You can be anything you want
            </h1>
            <p className="text-lg md:text-2xl text-purple-200/90 font-medium max-w-2xl mt-4 mb-12">
              Step into the future of real-time AI pose tracking and stylish visual transformations.
            </p>
            <Button 
              size="lg" 
              className="mt-8 text-lg px-8 py-8 rounded-full bg-white text-purple-950 hover:bg-purple-50 hover:scale-105 transition-all duration-300 shadow-[0_0_50px_-10px_rgba(255,255,255,0.4)] font-bold group"
              onClick={() => setStep('mode-select')}
            >
              Let's get started <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>
        ) : (
          <div className="w-full max-w-3xl animate-in slide-in-from-bottom-12 fade-in duration-700">
            <h2 className="text-5xl md:text-6xl font-black mb-12 drop-shadow-xl tracking-tight">Select mode.</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              <button 
                onClick={() => onModeSelect('webcam')}
                className="group relative overflow-hidden rounded-[2rem] bg-white/5 p-8 hover:bg-white/10 border border-white/10 transition-all duration-300 hover:-translate-y-2 hover:shadow-[0_20px_40px_-15px_rgba(168,85,247,0.4)] text-left flex flex-col items-start gap-6 backdrop-blur-md"
              >
                <div className="p-4 bg-purple-500/20 rounded-2xl group-hover:bg-purple-500/40 transition-colors shadow-inner">
                  <Camera className="w-10 h-10 text-purple-200" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold mb-3 text-white">Webcam mode</h3>
                  <p className="text-purple-200/80 leading-relaxed text-sm md:text-base">Experience the raw webcam feed with highly accurate, minimal overlay tracking metrics.</p>
                </div>
                <div className="mt-auto pt-4 flex items-center text-sm font-semibold text-purple-300 opacity-0 group-hover:opacity-100 transition-opacity -translate-x-4 group-hover:translate-x-0 duration-300">
                  Launch <ArrowRight className="ml-1 w-4 h-4" />
                </div>
              </button>

              <button 
                onClick={() => onModeSelect('wireframe')}
                className="group relative overflow-hidden rounded-[2rem] bg-white/5 p-8 hover:bg-white/10 border border-white/10 transition-all duration-300 hover:-translate-y-2 hover:shadow-[0_20px_40px_-15px_rgba(168,85,247,0.4)] text-left flex flex-col items-start gap-6 backdrop-blur-md"
              >
                <div className="p-4 bg-fuchsia-500/20 rounded-2xl group-hover:bg-fuchsia-500/40 transition-colors shadow-inner">
                  <Layers className="w-10 h-10 text-fuchsia-200" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold mb-3 text-white">Wireframe mode</h3>
                  <p className="text-purple-200/80 leading-relaxed text-sm md:text-base">Focus entirely on the data by isolating pose recognition with a full structural skeleton overlap.</p>
                </div>
                <div className="mt-auto pt-4 flex items-center text-sm font-semibold text-fuchsia-300 opacity-0 group-hover:opacity-100 transition-opacity -translate-x-4 group-hover:translate-x-0 duration-300">
                  Launch <ArrowRight className="ml-1 w-4 h-4" />
                </div>
              </button>

            </div>
          </div>
        )}
      </div>
    </div>
  )
}
