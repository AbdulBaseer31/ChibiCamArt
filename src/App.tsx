import { useState } from 'react'
import { Dashboard } from './components/Dashboard'
import { LandingPage } from './components/LandingPage'

type ViewState = 'landing' | 'dashboard'
type Mode = 'webcam' | 'wireframe'

function App() {
  const [view, setView] = useState<ViewState>('landing')
  const [selectedMode, setSelectedMode] = useState<Mode>('webcam')

  const handleModeSelect = (mode: Mode) => {
    setSelectedMode(mode);
    setView('dashboard');
  }

  if (view === 'landing') {
    return <LandingPage onModeSelect={handleModeSelect} />
  }

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30 animate-in fade-in duration-700">
      <Dashboard 
        initialMode={selectedMode} 
        onBack={() => setView('landing')} 
      />
    </div>
  )
}

export default App
