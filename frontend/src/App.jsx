import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Activity, LayoutGrid, CheckCircle, Circle, FileText, Download, Hexagon, X, Layers, MousePointer2, Maximize2, Search, Bell, Database, Archive, Settings, HelpCircle, Save, FileOutput, ZoomIn, ZoomOut, Eye, Edit3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const calculatePolygonArea = (points) => {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    let j = (i + 1) % points.length;
    area += points[i].x * points[j].y;
    area -= points[j].x * points[i].y;
  }
  return Math.abs(area / 2);
};

const calculatePolygonPerimeter = (points) => {
  let perimeter = 0;
  for (let i = 0; i < points.length; i++) {
    let j = (i + 1) % points.length;
    let dx = points[j].x - points[i].x;
    let dy = points[j].y - points[i].y;
    perimeter += Math.sqrt(dx * dx + dy * dy);
  }
  return perimeter;
};

const calculateCircularity = (area, perimeter) => {
  if (perimeter === 0) return 0;
  return (4 * Math.PI * area) / (perimeter * perimeter);
};

export default function App() {
  const [file, setFile] = useState(null);
  const [filename, setFilename] = useState('');
  const [preview, setPreview] = useState(null);
  const [tiles, setTiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [selectedTileName, setSelectedTileName] = useState(null);
  const [history, setHistory] = useState([]);
  
  const [viewHeatmap, setViewHeatmap] = useState(false);
  const [showConfidence, setShowConfidence] = useState(false);

  // Manual Annotation State
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [currentPath, setCurrentPath] = useState([]);
  const [manualMasks, setManualMasks] = useState({});
  const svgRef = useRef(null);

  // Canvas Pan/Zoom state
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });

  const activeTile = tiles.find(t => t.tile_name === selectedTileName);

  useEffect(() => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  }, [preview, results?.heatmap, viewHeatmap]);

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setFilename(selected.name);
      setResults(null);
      setTiles([]);
      setProgress(0);
      setError(null);
      setViewHeatmap(false);
      setManualMasks({});
      
      const formData = new FormData();
      formData.append("file", selected);
      try {
        const res = await fetch("http://localhost:8000/preview", { method: "POST", body: formData });
        if (res.ok) {
          const data = await res.json();
          setPreview(data.preview);
        }
      } catch (err) {
        console.error("Preview generation failed:", err);
      }
    }
  };

  const runInference = async () => {
    if (!file) return;
    
    if ("Notification" in window && Notification.permission !== "granted" && Notification.permission !== "denied") {
      Notification.requestPermission();
    }

    setLoading(true);
    setError(null);
    setTiles([]);
    setResults(null);
    setProgress(0);
    setViewHeatmap(false);
    setManualMasks({});

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_type", "unetplusplus");

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Inference failed");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      
      let currentTiles = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); 

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.type === 'progress') {
                setProgress(data.progress * 100);
                if (data.tile) {
                  currentTiles.push(data.tile);
                  setTiles([...currentTiles]);
                }
              } else if (data.type === 'final') {
                setResults(data);
                const sessionRecord = {
                  id: Date.now(),
                  filename: file.name,
                  model: 'unetplusplus',
                  cells: data.total_cells,
                  tiles: currentTiles,
                  timestamp: new Date().toLocaleTimeString(),
                  avgArea: data.average_size,
                  avgCirc: data.average_circularity
                };
                setHistory(prev => [sessionRecord, ...prev].slice(0, 10));
                
                if ("Notification" in window && Notification.permission === "granted") {
                  new Notification("Analysis Complete", {
                    body: `CHANA has finished processing ${file.name}. Detected ${data.total_cells} cells.`,
                    icon: "/vite.svg"
                  });
                }
              }
            } catch (e) {
              console.error("JSON parse error on line:", line, e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setProgress(100);
    }
  };

  const downloadCSV = (tilesData) => {
    if (!tilesData || tilesData.length === 0) return;
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "TileName,CellID,Area,Circularity,Manual\n";
    tilesData.forEach(t => {
      if(t.cells && t.cells.length > 0) {
        t.cells.forEach(c => {
          csvContent += `${t.tile_name},${c.id},${c.area.toFixed(2)},${c.circularity.toFixed(3)},${c.manual ? 'Yes' : 'No'}\n`;
        });
      }
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `chana_extraction_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Manual Annotation Logic
  const handleSvgMouseDown = (e) => {
    if (!isDrawingMode || !activeTile) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 512;
    const y = ((e.clientY - rect.top) / rect.height) * 512;
    setCurrentPath([{x, y}]);
  };

  const handleSvgMouseMove = (e) => {
    if (!isDrawingMode || currentPath.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 512;
    const y = ((e.clientY - rect.top) / rect.height) * 512;
    setCurrentPath(prev => [...prev, {x, y}]);
  };

  const handleSvgMouseUp = () => {
    if (!isDrawingMode || currentPath.length === 0) return;
    if (currentPath.length < 3) {
      setCurrentPath([]);
      return;
    }
    
    const area = calculatePolygonArea(currentPath);
    const perimeter = calculatePolygonPerimeter(currentPath);
    const circularity = calculateCircularity(area, perimeter);
    
    // Minimum realistic area check
    if (area < 5) {
       setCurrentPath([]);
       return;
    }
    
    const newCellId = activeTile.cells.length > 0 ? Math.max(...activeTile.cells.map(c => c.id)) + 1 : 1;
    const newCell = { id: newCellId, area, circularity, manual: true };
    
    const updatedTile = {
      ...activeTile,
      count: activeTile.count + 1,
      cells: [...activeTile.cells, newCell]
    };
    
    
    setTiles(prev => prev.map(t => t.tile_name === updatedTile.tile_name ? updatedTile : t));
    
    setResults(prev => {
      if (!prev) return prev;
      const newTotalCells = prev.total_cells + 1;
      const newAreas = [...prev.areas, area];
      const newCircs = [...prev.circularities, circularity];
      const newAvgArea = newAreas.reduce((a,b) => a+b, 0) / newAreas.length;
      const newAvgCirc = newCircs.reduce((a,b) => a+b, 0) / newCircs.length;
      return {
        ...prev,
        total_cells: newTotalCells,
        average_size: newAvgArea,
        average_circularity: newAvgCirc,
        areas: newAreas,
        circularities: newCircs
      };
    });
    
    setManualMasks(prev => ({
      ...prev,
      [activeTile.tile_name]: [...(prev[activeTile.tile_name] || []), currentPath]
    }));
    
    setCurrentPath([]);
  };

  let areaData = [];
  let circData = [];
  if (results && results.areas && results.circularities) {
    if (results.areas.length > 0) {
      const minArea = Math.min(...results.areas);
      const maxArea = Math.max(...results.areas);
      const areaStep = (maxArea - minArea) / 10 || 1;
      let aBins = new Array(10).fill(0);
      results.areas.forEach(a => {
        let idx = Math.floor((a - minArea) / areaStep);
        if (idx >= 10) idx = 9;
        aBins[idx]++;
      });
      areaData = aBins.map((count, i) => ({
        name: Math.round(minArea + (i * areaStep)).toString(),
        count
      }));
    }

    if (results.circularities.length > 0) {
      const minCirc = Math.min(...results.circularities);
      const maxCirc = Math.max(...results.circularities);
      const circStep = (maxCirc - minCirc) / 10 || 0.1;
      let cBins = new Array(10).fill(0);
      results.circularities.forEach(c => {
        let idx = Math.floor((c - minCirc) / circStep);
        if (idx >= 10) idx = 9;
        cBins[idx]++;
      });
      circData = cBins.map((count, i) => ({
        name: (minCirc + (i * circStep)).toFixed(2),
        count
      }));
    }
  }

  const handleWheel = (e) => {
    const scaleChange = e.deltaY * -0.001;
    setScale(s => Math.min(Math.max(0.1, s + scaleChange), 5));
  };
  const handlePanDown = (e) => {
    setIsDragging(true);
    dragStart.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };
  const handlePanMove = (e) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  };
  const handlePanUp = () => setIsDragging(false);

  return (
    <div className="flex h-screen w-screen bg-gradient-to-br from-[#fcf9fb] to-[#fce7f3] text-[#1a181a] font-sans overflow-hidden p-4 gap-4">
      
      {/* 1. Left Sidebar Navigation */}
      <nav className="w-[260px] bg-white/80 backdrop-blur-md rounded-3xl border border-rose-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col justify-between shrink-0 h-full z-10 overflow-hidden">
        <div className="flex flex-col h-full overflow-hidden">
          <div className="p-6 shrink-0">
            <h1 className="text-xl font-bold tracking-tight text-[#1a181a] flex items-center">
              <Hexagon size={22} className="mr-2 text-[#7b1738]" />
              CHANA
            </h1>
          </div>
          
          <div className="flex-1 overflow-y-auto custom-scrollbar px-4 pb-4">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3 pl-2">Session History</h3>
            <div className="space-y-3">
              {history.length === 0 ? (
                <div className="text-xs text-slate-400 italic text-center py-4">No past sessions</div>
              ) : (
                history.map(session => (
                  <div key={session.id} className="bg-white/80 backdrop-blur-md border border-rose-100 p-3 rounded-2xl border border-slate-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] relative group text-xs hover:border-rose-300 transition-colors">
                     <div className="font-bold text-slate-700 truncate pr-6" title={session.filename}>{session.filename}</div>
                     <div className="text-[10px] text-[#7b1738] font-semibold mb-1">{session.model}</div>
                     <div className="flex justify-between text-slate-500 mb-1">
                        <span>{session.cells} cells</span>
                        <span>{session.timestamp}</span>
                     </div>
                     <div className="flex justify-between text-slate-400 text-[10px]">
                        <span>Area: {session.avgArea ? session.avgArea.toFixed(1) : '---'} px²</span>
                        <span>Circ: {session.avgCirc ? session.avgCirc.toFixed(3) : '---'}</span>
                     </div>
                     <button onClick={() => downloadCSV(session.tiles)} className="absolute top-2 right-2 p-1.5 text-slate-400 hover:text-[#7b1738] hover:bg-rose-50 rounded transition-colors opacity-0 group-hover:opacity-100" title="Download CSV">
                       <Download size={14} />
                     </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 z-0 gap-4">
        
        {/* Header */}
        <header className="h-16 bg-white/80 backdrop-blur-md rounded-full border border-rose-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex items-center px-6 justify-between shrink-0">
          <div className="flex items-center space-x-4">
            <span className="text-xs font-bold text-slate-500 uppercase">Active Session:</span>
            <span className="text-sm font-bold text-[#1a181a] bg-slate-100 px-3 py-1 rounded-md">{filename || 'NONE'}</span>
          </div>
        </header>

        {/* Content Wrapper */}
        <div className="flex-1 flex gap-4 overflow-hidden">
          
          {/* 2. Control Panel (Left Pane) */}
          <aside className="w-[300px] bg-white/80 backdrop-blur-md rounded-3xl border border-rose-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col overflow-hidden shrink-0 custom-scrollbar">
            
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {/* Load Specimen */}
              <div className="border-2 border-dashed border-slate-100 rounded-3xl p-6 text-center bg-slate-50 relative group">
                <input type="file" id="file-upload" className="hidden" accept="image/*" onChange={handleFileChange} />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                  <div className="p-3 bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl mb-4 group-hover:border-rose-300 transition-colors shadow-[0_8px_30px_rgba(159,18,57,0.04)]">
                    <UploadCloud size={24} className="text-slate-400 group-hover:text-[#7b1738] transition-colors" />
                  </div>
                  <h3 className="font-bold text-[#1a181a] text-sm mb-1">Load Image</h3>
                  <p className="text-xs text-slate-400">Accepts .SVS, .TIFF</p>
                </label>
                {filename && (
                   <div className="mt-4 pt-4 border-t border-slate-100 flex justify-between items-center">
                      <span className="text-xs font-semibold text-slate-600 truncate max-w-[150px]">{filename}</span>
                      <button onClick={() => { setPreview(null); setFile(null); setResults(null); setTiles([]); }} className="text-[10px] bg-slate-200 px-2 py-1 rounded text-slate-600 hover:bg-slate-300">Clear</button>
                   </div>
                )}
              </div>

              {/* Heatmap Overlay */}
              {results?.heatmap && (
                <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl overflow-hidden shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col mb-4">
                  <div className="bg-slate-50 border-b border-slate-100 px-5 py-4">
                    <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Global Heatmap</h3>
                  </div>
                  <div className="p-4 bg-white/80 backdrop-blur-md border border-rose-100 flex justify-center">
                    <img src={results.heatmap} alt="Heatmap" className="w-full h-auto rounded-3xl object-contain border border-slate-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)]" />
                  </div>
                </div>
              )}
              
              <div className="pt-4 border-t border-slate-100 mt-auto">
                  <button 
                    onClick={runInference}
                    disabled={loading || !file}
                    className="w-full bg-[#1a181a] hover:bg-black text-white py-3.5 rounded-2xl font-bold text-sm shadow-lg shadow-slate-200 transition-all flex justify-center items-center disabled:opacity-50"
                  >
                    {loading ? <span className="flex items-center"><Activity size={16} className="mr-2 animate-spin"/> Processing...</span> : "Commence Analysis"}
                  </button>
                  {loading && (
                    <div className="mt-4 w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                      <div className="bg-[#7b1738] h-1.5 rounded-full transition-all duration-300" style={{ width: `${progress}%` }}></div>
                    </div>
                  )}
              </div>
            </div>
          </aside>


          {/* 3. Main Workspace Area */}
          <main className="flex-1 flex flex-col min-w-0 overflow-hidden gap-4">
            
            {/* Top: Interactive Canvas */}
            <div className="flex-1 bg-[#e2e8f0] border border-slate-300 rounded-3xl overflow-hidden relative shadow-inner flex items-center justify-center cursor-grab active:cursor-grabbing"
                 onWheel={handleWheel} onMouseDown={handlePanDown} onMouseMove={handlePanMove} onMouseUp={handlePanUp} onMouseLeave={handlePanUp}>
              
              {!preview ? (
                <div className="flex flex-col items-center text-slate-400">
                  <Eye size={48} className="mb-4 opacity-50" />
                  <p className="font-semibold">Canvas Awaiting Specimen</p>
                </div>
              ) : (
                <div style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`, transition: isDragging ? 'none' : 'transform 0.1s ease-out' }} className="origin-center w-full h-full flex items-center justify-center">
                  <img src={viewHeatmap && results?.heatmap ? results.heatmap : preview} alt="Specimen" className="max-w-none shadow-2xl pointer-events-none" draggable="false" />
                </div>
              )}
              
              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col space-y-2">
                <button onClick={(e) => { e.stopPropagation(); setScale(s => s + 0.2); }} className="p-2 bg-[#1a181a]/80 text-white rounded shadow hover:bg-slate-700 transition"><ZoomIn size={16}/></button>
                <button onClick={(e) => { e.stopPropagation(); setScale(s => Math.max(0.1, s - 0.2)); }} className="p-2 bg-[#1a181a]/80 text-white rounded shadow hover:bg-slate-700 transition"><ZoomOut size={16}/></button>
                <button onClick={(e) => { e.stopPropagation(); setScale(1); setPan({x:0, y:0}); }} className="p-2 bg-[#1a181a]/80 text-white rounded shadow hover:bg-slate-700 transition" title="Fit to Screen"><Maximize2 size={16}/></button>
                {results?.heatmap && (
                   <button onClick={(e) => { e.stopPropagation(); setViewHeatmap(!viewHeatmap); }} className={`p-2 rounded shadow transition ${viewHeatmap ? 'bg-[#7b1738] text-white' : 'bg-[#1a181a]/80 text-white hover:bg-slate-700'}`} title="Toggle Heatmap"><Layers size={16}/></button>
                )}
              </div>
              
              {loading && <div className="absolute bottom-0 left-0 h-1 bg-[#7b1738] transition-all duration-300" style={{ width: `${progress}%` }}></div>}
            </div>

            {/* Middle: Clinical Metrics Blocks */}
            <div className="grid grid-cols-3 gap-4 shrink-0">
              <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl p-5 shadow-[0_8px_30px_rgba(159,18,57,0.04)]">
                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Detections</h4>
                <div className="flex items-end justify-between">
                  <div>
                    <span className="text-3xl font-bold text-[#1a181a]">{results ? results.total_cells.toLocaleString() : '---'}</span>
                    <span className="text-sm font-semibold text-slate-400 ml-1">cells</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl p-5 shadow-[0_8px_30px_rgba(159,18,57,0.04)]">
                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Avg. Area</h4>
                <div className="flex items-end justify-between">
                  <div>
                    <span className="text-3xl font-bold text-[#1a181a]">{results ? results.average_size.toFixed(1) : '---'}</span>
                    <span className="text-sm font-semibold text-slate-400 ml-1">px²</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl p-5 shadow-[0_8px_30px_rgba(159,18,57,0.04)]">
                <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Cellular Circularity</h4>
                <div className="flex items-end justify-between">
                  <div>
                    <span className="text-3xl font-bold text-[#1a181a]">{results ? results.average_circularity.toFixed(3) : '---'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom: Two Morphology Distributions */}
            <div className="grid grid-cols-2 gap-4 h-48 shrink-0">
              <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl p-5 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col h-full">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-[10px] font-bold text-[#1a181a] uppercase tracking-wider">Area Distribution (px²)</h4>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={areaData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} contentStyle={{borderRadius: '4px', border: '1px solid #e2e8f0', fontSize: '11px'}} />
                      <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                        {areaData.map((entry, index) => {
                           const maxVal = Math.max(...areaData.map(d => d.count));
                           return <Cell key={`cell-${index}`} fill={entry.count === maxVal ? '#7b1738' : '#e2e8f0'} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-3xl p-5 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col h-full">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-[10px] font-bold text-[#1a181a] uppercase tracking-wider">Circularity Distribution</h4>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={circData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                      <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{fill: 'rgba(0,0,0,0.02)'}} contentStyle={{borderRadius: '4px', border: '1px solid #e2e8f0', fontSize: '11px'}} />
                      <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                        {circData.map((entry, index) => {
                           const maxVal = Math.max(...circData.map(d => d.count));
                           return <Cell key={`cell-${index}`} fill={entry.count === maxVal ? '#7b1738' : '#e2e8f0'} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

          </main>

          {/* 4. Right Edge: Extraction Tile Library */}
          <aside className="w-[320px] bg-white/80 backdrop-blur-xl rounded-3xl border border-rose-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] flex flex-col shrink-0 overflow-hidden">
            <div className="flex-1 flex flex-col overflow-hidden bg-slate-50/30">
               <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-white/80 backdrop-blur-md border border-rose-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] z-10">
                  <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Extraction Library</h3>
                  {results && (
                     <button onClick={() => downloadCSV(tiles)} className="text-xs font-bold text-[#7b1738] flex items-center hover:text-teal-900 transition-colors">
                        <Download size={12} className="mr-1" /> Export CSV
                     </button>
                  )}
               </div>
               
               <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                  {tiles.length === 0 ? (
                     <div className="text-xs text-slate-400 italic text-center py-8">Library empty</div>
                  ) : (
                     <div className="grid grid-cols-2 gap-3">
                        {tiles.map((tile, i) => (
                           <div key={i} onClick={() => { setSelectedTileName(tile.tile_name); setIsDrawingMode(false); setCurrentPath([]); }} className="bg-white/80 backdrop-blur-md border border-rose-100 border border-slate-100 rounded-2xl p-2 cursor-pointer hover:border-rose-300 hover:shadow-md transition-all group">
                              <div className="aspect-square bg-slate-100 rounded-md overflow-hidden relative mb-2">
                                 <img src={showConfidence && tile.confidence ? tile.confidence : tile.mask} alt={`Tile ${i}`} className="w-full h-full object-cover" />
                                 {/* Render any manual masks over the thumbnail as well */}
                                 {(manualMasks[tile.tile_name] || []).length > 0 && (
                                   <svg viewBox="0 0 512 512" className="absolute inset-0 w-full h-full pointer-events-none">
                                     {(manualMasks[tile.tile_name] || []).map((path, idx) => (
                                       <polygon key={idx} points={path.map(p => `${p.x},${p.y}`).join(' ')} fill="rgba(168, 85, 247, 0.6)" stroke="#be123c" strokeWidth="4" />
                                     ))}
                                   </svg>
                                 )}
                              </div>
                              <div className="text-[10px] font-bold text-slate-700 mb-0.5 truncate">{tile.tile_name}</div>
                              <div className="text-[9px] font-semibold text-[#7b1738]">{tile.count} cells detected</div>
                           </div>
                        ))}
                     </div>
                  )}
               </div>
               
               {tiles.length > 0 && (
                  <div className="p-3 bg-white/80 backdrop-blur-md border border-rose-100 border-t border-slate-100 flex items-center justify-between">
                     <span className="text-xs font-bold text-slate-600">Saliency Overlay</span>
                     <button onClick={() => setShowConfidence(!showConfidence)} className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${showConfidence ? 'bg-[#7b1738]' : 'bg-slate-300'}`}>
                        <span className={`inline-block h-3 w-3 transform rounded-full bg-white/80 backdrop-blur-md border border-rose-100 transition-transform ${showConfidence ? 'translate-x-5' : 'translate-x-1'}`} />
                     </button>
                  </div>
               )}
            </div>
          </aside>
        </div>
      </div>

      {/* Interactive Tile Modal */}
      {activeTile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#7b1738]/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white/80 backdrop-blur-md border border-rose-100 rounded-2xl shadow-2xl w-full max-w-4xl overflow-hidden flex max-h-[85vh]">
            
            {/* Modal Image Left Side */}
            <div className="w-1/2 bg-slate-50 p-6 flex flex-col items-center justify-center relative border-r border-slate-100">
              
              <div className="absolute top-4 right-4 flex bg-white/80 backdrop-blur-md border border-rose-100 rounded-2xl shadow-[0_8px_30px_rgba(159,18,57,0.04)] border border-slate-100 overflow-hidden z-20">
                 <button onClick={() => setShowConfidence(false)} className={`px-3 py-1.5 text-[10px] font-bold ${!showConfidence ? 'bg-[#7b1738] text-white' : 'text-slate-600 hover:bg-slate-50'}`}>Mask</button>
                 <button onClick={() => setShowConfidence(true)} className={`px-3 py-1.5 text-[10px] font-bold ${showConfidence ? 'bg-[#7b1738] text-white' : 'text-slate-600 hover:bg-slate-50'}`}>Saliency</button>
                 <button onClick={() => setIsDrawingMode(!isDrawingMode)} className={`px-3 py-1.5 text-[10px] font-bold flex items-center ${isDrawingMode ? 'bg-fuchsia-600 text-white' : 'text-slate-600 hover:bg-slate-50'}`} title="Draw Correction">
                    <Edit3 size={12} className="mr-1"/> Annotate
                 </button>
              </div>

              {/* The Tile Image with SVG Drawing Overlay */}
              <div className="relative aspect-square w-full max-h-[65vh] flex items-center justify-center shadow-md rounded-2xl overflow-hidden border border-slate-200 bg-white/80 backdrop-blur-md border border-rose-100 group">
                <img src={showConfidence && activeTile.confidence ? activeTile.confidence : activeTile.mask} alt="Tile detail" className="absolute inset-0 w-full h-full object-contain pointer-events-none" />
                
                <svg 
                  ref={svgRef}
                  viewBox="0 0 512 512" 
                  className={`absolute inset-0 w-full h-full ${isDrawingMode ? 'cursor-crosshair' : 'pointer-events-none'}`}
                  onMouseDown={handleSvgMouseDown}
                  onMouseMove={handleSvgMouseMove}
                  onMouseUp={handleSvgMouseUp}
                  onMouseLeave={handleSvgMouseUp}
                >
                  {(manualMasks[activeTile.tile_name] || []).map((path, idx) => (
                    <polygon 
                      key={idx}
                      points={path.map(p => `${p.x},${p.y}`).join(' ')}
                      fill="rgba(217, 70, 239, 0.4)" 
                      stroke="#c026d3" 
                      strokeWidth="2"
                    />
                  ))}
                  
                  {currentPath.length > 0 && (
                    <polyline 
                      points={currentPath.map(p => `${p.x},${p.y}`).join(' ')}
                      fill="none" 
                      stroke="#ec4899" 
                      strokeWidth="3"
                      strokeDasharray="4"
                    />
                  )}
                </svg>

                {isDrawingMode && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-900/80 text-white text-[10px] font-bold px-3 py-1.5 rounded-full pointer-events-none">
                     Click and drag to mask a cell
                  </div>
                )}
              </div>
            </div>
            
            {/* Modal Table Right Side */}
            <div className="w-1/2 flex flex-col bg-white/80 backdrop-blur-md border border-rose-100">
              <div className="p-5 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                <div>
                  <h3 className="text-lg font-bold text-[#1a181a]">{activeTile.tile_name}</h3>
                  <p className="text-xs font-semibold text-[#7b1738] mt-1">{activeTile.count} morphometric detections</p>
                </div>
                <button onClick={() => { setSelectedTileName(null); setIsDrawingMode(false); setCurrentPath([]); }} className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-200 rounded-full transition-colors">
                  <X size={20} />
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto custom-scrollbar">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 sticky top-0 border-b border-slate-100 shadow-[0_8px_30px_rgba(159,18,57,0.04)] z-10">
                    <tr>
                      <th className="px-5 py-3 font-bold text-slate-500 uppercase tracking-wider">ID</th>
                      <th className="px-5 py-3 font-bold text-slate-500 uppercase tracking-wider">Area (px²)</th>
                      <th className="px-5 py-3 font-bold text-slate-500 uppercase tracking-wider">Circularity</th>
                      <th className="px-5 py-3 font-bold text-slate-500 uppercase tracking-wider">Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {activeTile.cells.map((cell, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition-colors">
                        <td className="px-5 py-3 font-mono font-bold text-[#7b1738]">#{cell.id}</td>
                        <td className="px-5 py-3 font-semibold text-slate-700">{cell.area.toFixed(1)}</td>
                        <td className="px-5 py-3 font-semibold text-slate-700">{cell.circularity.toFixed(3)}</td>
                        <td className="px-5 py-3 font-semibold">
                          {cell.manual ? <span className="text-fuchsia-700 bg-pink-50 px-2 py-0.5 rounded text-[9px] font-bold">Manual</span> : <span className="text-slate-400">AI</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
