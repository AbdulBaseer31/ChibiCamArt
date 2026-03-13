import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import { Switch } from './ui/switch'
import { Label } from './ui/label'
import { Slider } from './ui/slider'
import { Button } from './ui/button'
import { Video, Settings2, Activity, Play, Square, ChevronLeft } from 'lucide-react'

type Mode = 'webcam' | 'wireframe' | 'matrix' | 'glitch' | 'terminal' | 'hologram' | 'dot_field'

interface DashboardProps {
  initialMode?: Mode;
  onBack?: () => void;
}

export function Dashboard({ initialMode = 'webcam', onBack }: DashboardProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [confidence, setConfidence] = useState([50])
  const [useFullModel, setUseFullModel] = useState(false)
  const [showWireframe, setShowWireframe] = useState(initialMode === 'wireframe')
  const [applyStylize, setApplyStylize] = useState(false)
  const [stylizeMode, setStylizeMode] = useState<'face' | 'full' | 'anime_cv' | 'ghibli_cv' | 'watercolor_cv'>('anime_cv')
  const [currentViewMode, setCurrentViewMode] = useState<Mode>(initialMode)

  // WebSocket State
  const [frameData, setFrameData] = useState<string | null>(null)
  const [sysStats, setSysStats] = useState({
    fps: 0,
    latency_ms: 0,
    camera_active: false,
    model_name: 'MediaPipe v0.10.32',
    detected_entities: 0
  })

  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (isRunning) {
      wsRef.current = new WebSocket('ws://localhost:8000/ws')

      wsRef.current.onopen = () => {
        wsRef.current?.send(JSON.stringify({ 
          type: 'settings', 
          showWireframe,
          viewMode: currentViewMode,
          applyStylize,
          stylizeMode
        }))
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'frame') {
            setFrameData(data.image)
            setSysStats(prev => ({
              ...prev,
              fps: data.fps ?? prev.fps,
              latency_ms: data.latency_ms ?? prev.latency_ms,
              camera_active: data.camera_active ?? prev.camera_active,
              model_name: data.model_name ?? prev.model_name,
              detected_entities: data.detected_entities ?? prev.detected_entities
            }))
          }
        } catch (e) {
          console.error("Error parsing websocket message", e)
        }
      }

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error)
        setIsRunning(false)
      }

      wsRef.current.onclose = () => {
        setFrameData(null)
        setSysStats(prev => ({ ...prev, camera_active: false }))
      }
    } else {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      setFrameData(null)
      setSysStats(prev => ({ ...prev, camera_active: false }))
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [isRunning])

  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        type: 'settings', 
        showWireframe,
        viewMode: currentViewMode,
        applyStylize,
        stylizeMode
      }))
    }
  }, [showWireframe, initialMode, applyStylize, stylizeMode])

  return (
    <div className="container mx-auto p-4 min-h-screen flex flex-col gap-6">
      <header className="flex justify-between items-center py-4 border-b border-border/50">
        <div className="flex items-center gap-3">
          {onBack && (
            <Button variant="ghost" size="icon" onClick={onBack} className="mr-2 rounded-full hover:bg-accent/50">
              <ChevronLeft className="w-5 h-5" />
            </Button>
          )}
          <div className="p-2 bg-primary/20 rounded-lg">
            <Activity className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/50 bg-clip-text text-transparent">
            ChibiCam Dashboard
          </h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-secondary text-sm">
            <span className={`w-2 h-2 rounded-full ${isRunning && sysStats.camera_active ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
            {isRunning && sysStats.camera_active ? 'System Active' : 'System Ready'}
          </div>
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        {/* Main Video View */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <Card className="flex-1 overflow-hidden group">
            <div className="relative w-full h-full min-h-[500px] bg-black/40 flex items-center justify-center">
              {isRunning && frameData ? (
                 <img src={frameData} alt="Pose Tracking Stream" className="w-full h-full object-contain" />
              ) : isRunning ? (
                <div className="absolute inset-0 flex items-center justify-center text-muted-foreground flex-col gap-4 bg-black/60 backdrop-blur-sm">
                   <Video className="w-16 h-16 animate-pulse-slow text-primary/50" />
                   <p className="tracking-widest uppercase text-sm font-semibold">Connecting to Engine...</p>
                </div>
              ) : (
                <div className="flex flex-col items-center text-muted-foreground gap-4">
                  <Video className="w-16 h-16 opacity-50" />
                  <p>Camera feed is inactive</p>
                </div>
              )}
              
              {/* Overlay stats that show on hover or when running */}
              {isRunning && sysStats.camera_active && (
                <div className="absolute top-4 left-4 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="px-2 py-1 bg-black/60 rounded text-xs font-mono text-primary">FPS: {sysStats.fps}</div>
                  <div className="px-2 py-1 bg-black/60 rounded text-xs font-mono text-emerald-400">Latency: {sysStats.latency_ms}ms</div>
                </div>
              )}
            </div>
          </Card>

          <div className="flex justify-center gap-4">
            <Button 
              size="lg" 
              variant={isRunning ? "destructive" : "default"}
              onClick={() => setIsRunning(!isRunning)}
              className="w-48 font-semibold tracking-wide"
            >
              {isRunning ? (
                <><Square className="w-4 h-4 mr-2" /> Stop Tracking</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> Start Tracking</>
              )}
            </Button>
          </div>
        </div>

        {/* Control Panel */}
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="text-lg flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-primary" />
                Tracker Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-6">
              
              <div className="flex items-center justify-between">
                <Label htmlFor="model-toggle" className="flex flex-col gap-1 cursor-pointer">
                  <span>Full Engine Model</span>
                  <span className="font-normal text-xs text-muted-foreground">High accuracy, higher latency</span>
                </Label>
                <Switch 
                  id="model-toggle" 
                  checked={useFullModel}
                  onCheckedChange={setUseFullModel}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <Label htmlFor="wireframe-toggle" className="flex flex-col gap-1 cursor-pointer">
                  <span>Draw Wireframe</span>
                  <span className="font-normal text-xs text-muted-foreground">Overlay skeletal pose data</span>
                </Label>
                <Switch 
                  id="wireframe-toggle" 
                  checked={showWireframe}
                  onCheckedChange={setShowWireframe}
                />
              </div>

              <div className="flex items-center justify-between border-t border-border/50 pt-4">
                <Label htmlFor="stylize-toggle" className="flex flex-col gap-1 cursor-pointer">
                  <span className="text-primary font-medium">✨ Artistic Effects</span>
                  <span className="font-normal text-xs text-muted-foreground">Select a visual style or filter</span>
                </Label>
                <Switch 
                  id="stylize-toggle" 
                  checked={applyStylize}
                  onCheckedChange={setApplyStylize}
                />
              </div>
              
              {applyStylize && (
                <div className="flex flex-col gap-2 border border-border/50 p-2 rounded-lg bg-muted/20">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${currentViewMode === 'matrix' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('matrix')
                      }}
                    >
                      Matrix
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${currentViewMode === 'glitch' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('glitch')
                      }}
                    >
                      Glitch
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${currentViewMode === 'terminal' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('terminal')
                      }}
                    >
                      Terminal
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${currentViewMode === 'hologram' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('hologram')
                      }}
                    >
                      Hologram
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${currentViewMode === 'dot_field' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('dot_field')
                      }}
                    >
                      Dot Field
                    </button>
                  </div>

                  <div className="text-xs font-semibold text-muted-foreground mt-2 mb-1 uppercase tracking-wider">Fast Filters (CPU)</div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${stylizeMode === 'anime_cv' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('webcam')
                        setStylizeMode('anime_cv')
                      }}
                    >
                      Anime
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${stylizeMode === 'ghibli_cv' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('webcam')
                        setStylizeMode('ghibli_cv')
                      }}
                    >
                      Ghibli
                    </button>
                    <button 
                      className={`py-1.5 px-2 rounded-md text-center transition-colors ${stylizeMode === 'watercolor_cv' ? 'bg-primary/20 text-primary font-medium border border-primary/30' : 'bg-muted/30 hover:bg-accent/50 text-muted-foreground border border-transparent'}`}
                      onClick={() => {
                        setCurrentViewMode('webcam')
                        setStylizeMode('watercolor_cv')
                      }}
                    >
                      Watercolor
                    </button>
                  </div>
                </div>
              )}

              <div className="space-y-4 pt-4 border-t border-border/50">
                <div className="flex justify-between items-center">
                  <Label>Confidence Threshold</Label>
                  <span className="text-xs font-mono bg-secondary px-2 py-1 rounded w-12 text-center">
                    {(confidence[0] / 100).toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={confidence}
                  onValueChange={setConfidence}
                  max={100}
                  step={1}
                  className="py-4"
                />
              </div>
            </CardContent>
          </Card>

          <Card className="flex-1">
             <CardHeader className="pb-4">
                <CardTitle className="text-lg">System Status</CardTitle>
             </CardHeader>
             <CardContent>
                <div className="space-y-4 text-sm text-muted-foreground">
                  <div className="flex justify-between border-b border-border/50 pb-2">
                    <span>Tracker Engine</span>
                    <span className="text-foreground">MediaPipe v0.10.32</span>
                  </div>
                  <div className="flex justify-between border-b border-border/50 pb-2">
                    <span>Active Camera</span>
                    <span className="text-foreground">None</span>
                  </div>
                  <div className="flex justify-between pb-2">
                    <span>Detected Entities</span>
                    <span className="text-foreground font-mono">0</span>
                  </div>
                </div>
             </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
