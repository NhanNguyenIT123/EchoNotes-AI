import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, Video, FileText, Cpu, Eye, MessageSquare,
  RefreshCw, Upload, Sparkles, Download, FileDown, Search,
  PanelLeftClose, PanelLeftOpen, Maximize2, Minimize2
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

// Simple Markdown Renderer component to avoid third-party HTML/Vite build friction
function SimpleMarkdown({ markdown }: { markdown: string }) {
  if (!markdown) return <p style={{ color: 'var(--text-muted)' }}>No notes generated yet.</p>;

  const lines = markdown.split('\n');
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];

  lines.forEach((line, index) => {
    // Code block check
    if (line.trim().startsWith('```')) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`code-${index}`} style={{ margin: '1rem 0' }}>
            <code>{codeBlockLines.join('\n')}</code>
          </pre>
        );
        codeBlockLines = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      return;
    }

    // Markdown Image check. Supports backend-served keyframe URLs and data URIs.
    const imgRegex = /^!\[([^\]]*)\]\((.+)\)$/;
    const imgMatch = line.match(imgRegex);
    if (imgMatch) {
      const imageSrc = imgMatch[2].trim();
      elements.push(
        <img 
          key={`img-${index}`} 
          src={imageSrc} 
          alt={imgMatch[1] || 'Keyframe'} 
          className="report-img"
          onError={(e) => { e.currentTarget.style.display = 'none'; }}
        />
      );
      return;
    }

    // Headings
    if (line.startsWith('# ')) {
      elements.push(<h1 key={index}>{line.substring(2)}</h1>);
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={index}>{line.substring(3)}</h2>);
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={index}>{line.substring(4)}</h3>);
    } else if (line.startsWith('#### ')) {
      elements.push(<h4 key={index}>{line.substring(5)}</h4>);
    }
    // Blockquote
    else if (line.startsWith('> ')) {
      elements.push(<blockquote key={index} style={{ borderLeft: '3px solid var(--blue-normal)', paddingLeft: '1rem', color: 'var(--text-secondary)', margin: '1rem 0', fontStyle: 'italic' }}>{line.substring(2)}</blockquote>);
    }
    // Bullets
    else if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      elements.push(<li key={index} style={{ marginLeft: '1.5rem', marginTop: '0.25rem', marginBottom: '0.25rem' }}>{line.trim().substring(2)}</li>);
    }
    // Divider
    else if (line.trim() === '---') {
      elements.push(<hr key={index} style={{ border: 'none', borderBottom: '1px solid var(--border-color)', margin: '1.5rem 0' }} />);
    }
    // Paragraph / Normal line
    else if (line.trim()) {
      elements.push(<p key={index} style={{ marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>{line}</p>);
    }
  });

  return <div className="report-markdown-preview">{elements}</div>;
}

export default function App() {
  // Sidebar states
  const [sourceMode, setSourceMode] = useState<'upload' | 'teams' | 'demo'>('upload');
  const [whisperModel, setWhisperModel] = useState('small');
  const [llmModel, setLlmModel] = useState('qwen2.5:7b-instruct');
  const [visionModel, setVisionModel] = useState('llava:7b');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [spokenLanguage, setSpokenLanguage] = useState('en');
  const [asrPrompt, setAsrPrompt] = useState('English technical lecture. Preserve exact English terms.');
  const [hotwords, setHotwords] = useState('');
  const [visualMode, setVisualMode] = useState('Fast: capture keyframes, no OCR');
  const [ssimSensitivity, setSsimSensitivity] = useState(0.94);
  const [minKeyframeGap, setMinKeyframeGap] = useState(20);
  const [maxKeyframes, setMaxKeyframes] = useState(80);
  const [frameCheckInterval, setFrameCheckInterval] = useState(10);
  const [acousticToggle, setAcousticToggle] = useState(false);
  const [useOsGlossary, setUseOsGlossary] = useState(true);

  // Teams Link states
  const [teamsUrl, setTeamsUrl] = useState('');
  const [deviceFlow, setDeviceFlow] = useState<{ user_code: string; verification_uri: string } | null>(null);
  const [teamsTokenReady, setTeamsTokenReady] = useState(false);

  // Upload state management
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [transcriptFile, setTranscriptFile] = useState<File | null>(null);
  const [uploadStatusText, setUploadStatusText] = useState('');
  const [projects, setProjects] = useState<any[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');

  // Pipeline execution state
  const [systemStatus, setSystemStatus] = useState({
    status: 'idle',
    stage: 'Awaiting inputs',
    progress: 0,
    error: null as string | null,
    active_video_name: null as string | null,
    has_transcript_upload: false,
    has_teams_video: false,
    generating_report: false,
    report_started_at: null as number | null,
    dataset_status: 'idle',
    dataset_progress: 0,
    active_project_id: null as string | null,
    database: { connected: false, error: null as string | null }
  });

  // Multimodal outputs
  const [activeTab, setActiveTab] = useState<'notes' | 'visual' | 'playback' | 'chat'>('notes');
  const [transcript, setTranscript] = useState<any[]>([]);
  const [slides, setSlides] = useState<any[]>([]);
  const [topicBlocks, setTopicBlocks] = useState<any[]>([]);
  const [reportMarkdown, setReportMarkdown] = useState('');
  const [synthesisMethod, setSynthesisMethod] = useState<'fast' | 'ollama'>('fast');
  const [searchQuery, setSearchQuery] = useState('');
  const [playbackTime, setPlaybackTime] = useState(0);
  const [activeTranscriptIndex, setActiveTranscriptIndex] = useState<number | null>(null);
  const [pendingSeekTime, setPendingSeekTime] = useState<number | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [focusMode, setFocusMode] = useState(false);

  // Grounded Chat state
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  // Ref handles
  const videoRef = useRef<HTMLVideoElement>(null);
  const activeTranscriptRef = useRef<HTMLDivElement | null>(null);

  // Fetch initial Ollama models list & status
  useEffect(() => {
    axios.get(`${API_BASE}/ollama-models`)
      .then(res => {
        if (res.data.models && res.data.models.length > 0) {
          setOllamaModels(res.data.models);
          // Set default to qwen if available, or first model
          if (res.data.models.includes('qwen2.5:7b-instruct')) {
            setLlmModel('qwen2.5:7b-instruct');
          } else {
            setLlmModel(res.data.models[0]);
          }
          const visionCandidate = res.data.models.find((model: string) => {
            const name = model.toLowerCase();
            return name.includes('llava') || name.includes('qwen') && name.includes('vl') || name.includes('minicpm-v') || name.includes('moondream');
          });
          if (visionCandidate) setVisionModel(visionCandidate);
        }
      })
      .catch(() => {});

    pollStatus();
    fetchProjects();
    // Poll system status every 1200ms
    const interval = setInterval(pollStatus, 1200);
    return () => clearInterval(interval);
  }, []);

  // Fetch results when status completes
  useEffect(() => {
    if (systemStatus.status === 'completed' && !systemStatus.generating_report) {
      fetchResults();
    }
  }, [systemStatus.status, systemStatus.generating_report]);

  useEffect(() => {
    if (!systemStatus.generating_report) return;

    const interval = setInterval(async () => {
      try {
        const reportRes = await axios.get(`${API_BASE}/report/markdown`);
        if (reportRes.data.markdown) {
          setReportMarkdown(reportRes.data.markdown);
        }
      } catch {
        // Keep polling status; report markdown can be empty during the first model call.
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [systemStatus.generating_report]);

  const pollStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/status`);
      setSystemStatus(res.data);
    } catch (e) {}
  };

  const fetchProjects = async () => {
    try {
      const res = await axios.get(`${API_BASE}/projects`);
      setProjects(res.data.projects || []);
      if (!selectedProjectId && res.data.projects?.length) {
        setSelectedProjectId(res.data.projects[0].id);
      }
    } catch {
      setProjects([]);
    }
  };

  const fetchResults = async () => {
    try {
      const resultsRes = await axios.get(`${API_BASE}/results`);
      setTranscript(resultsRes.data.transcript);
      setSlides(resultsRes.data.slides);
      setTopicBlocks(resultsRes.data.topic_blocks || []);

      const reportRes = await axios.get(`${API_BASE}/report/markdown`);
      setReportMarkdown(reportRes.data.markdown);
    } catch (e) {}
  };

  // --- ACTIONS ---

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setVideoFile(file);
    setUploadStatusText(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_BASE}/upload/video`, formData);
      setUploadStatusText(`Video ${file.name} uploaded successfully.`);
      pollStatus();
      fetchProjects();
    } catch (err: any) {
      setUploadStatusText(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleTranscriptUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setTranscriptFile(file);
    setUploadStatusText(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_BASE}/upload/transcript`, formData);
      setUploadStatusText(`Transcript ${file.name} uploaded successfully.`);
      pollStatus();
    } catch (err: any) {
      setUploadStatusText(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  // Microsoft Graph sign-in creation
  const handleMicrosoftSignIn = async () => {
    try {
      const res = await axios.post(`${API_BASE}/teams/device-code`);
      setDeviceFlow(res.data);
    } catch (err: any) {
      alert(`Sign in flow failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleCompleteSignIn = async () => {
    try {
      await axios.post(`${API_BASE}/teams/complete-login`);
      setTeamsTokenReady(true);
      setDeviceFlow(null);
      alert('Microsoft Graph sign-in completed successfully.');
    } catch (err: any) {
      alert(`Verification check: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleTeamsDownload = async () => {
    if (!teamsUrl.trim()) return;
    try {
      await axios.post(`${API_BASE}/teams/download`, { url: teamsUrl });
      pollStatus();
    } catch (err: any) {
      alert(`Download trigger failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleDemoLaunch = async () => {
    try {
      await axios.get(`${API_BASE}/demo-init`);
      pollStatus();
    } catch (err: any) {
      alert(`Demo setup failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleRunAnalysis = async () => {
    try {
      await axios.post(`${API_BASE}/analyze`, {
        whisper_model: whisperModel,
        selected_llm: llmModel,
        speech_language: spokenLanguage,
        lecture_profile: useOsGlossary ? 'Operating systems / inode-block-pointer' : 'General / no forced terminology',
        whisper_prompt: asrPrompt,
        whisper_hotwords: hotwords,
        use_os_glossary: useOsGlossary,
        vision_mode: visualMode,
        vision_model: visionModel,
        ssim_thresh: ssimSensitivity,
        min_slide_gap: minKeyframeGap,
        max_slide_count: maxKeyframes,
        frame_sample_interval: frameCheckInterval,
        analyze_acoustics_enabled: acousticToggle
      });
      pollStatus();
      fetchProjects();
    } catch (err: any) {
      alert(`Analysis failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleClearCache = async () => {
    if (!window.confirm('Are you sure you want to clear caches and start a new session?')) return;
    try {
      await axios.post(`${API_BASE}/clear`);
      setVideoFile(null);
      setTranscriptFile(null);
      setUploadStatusText('');
      setTranscript([]);
      setSlides([]);
      setTopicBlocks([]);
      setReportMarkdown('');
      setChatHistory([]);
      pollStatus();
      fetchProjects();
    } catch (err: any) {
      alert(`Clear failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleLoadProject = async () => {
    if (!selectedProjectId) return;
    try {
      const res = await axios.post(`${API_BASE}/projects/${selectedProjectId}/load`);
      setTranscript(res.data.project.transcript || []);
      setSlides(res.data.project.slides || []);
      setTopicBlocks(res.data.project.topic_blocks || []);
      setReportMarkdown(res.data.project.report_markdown || '');
      setChatHistory((res.data.chat || []).map((m: any) => ({ role: m.role, content: m.content })));
      pollStatus();
      fetchProjects();
    } catch (err: any) {
      alert(`Load project failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleSaveCurrentProject = async () => {
    try {
      await axios.post(`${API_BASE}/projects/save-current`);
      fetchProjects();
      alert('Current lecture project saved to PostgreSQL.');
    } catch (err: any) {
      alert(`Save project failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  // --- REPORT NOTES ACTIONS ---

  const handleGenerateReport = async () => {
    try {
      await axios.post(`${API_BASE}/report/generate`, {
        method: synthesisMethod,
        model_name: llmModel
      });
      pollStatus();
    } catch (err: any) {
      alert(`Failed to trigger report build: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleResetReportGeneration = async () => {
    try {
      await axios.post(`${API_BASE}/report/reset`);
      pollStatus();
      fetchResults();
    } catch (err: any) {
      alert(`Failed to reset synthesis state: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleDownloadMarkdown = () => {
    const blob = new Blob([reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'EchoNotes_SmartNotes.md';
    a.click();
  };

  const handleDownloadPDF = () => {
    window.open(`${API_BASE}/report/pdf`, '_blank');
  };

  // --- PLAYBACK TIMELINE SEEKING ---
  const getSegmentEnd = (seg: any, idx: number, list: any[]) => {
    const start = Number(seg?.start) || 0;
    const explicitEnd = Number(seg?.end);
    if (Number.isFinite(explicitEnd) && explicitEnd > start) return explicitEnd;

    const nextStart = Number(list[idx + 1]?.start);
    if (Number.isFinite(nextStart) && nextStart > start) return nextStart;

    return start + 4;
  };

  const findActiveTranscriptIndex = (time: number) => {
    return transcript.findIndex((seg, idx) => {
      const start = Number(seg?.start) || 0;
      const end = getSegmentEnd(seg, idx, transcript);
      return time >= start && time < end;
    });
  };

  const handleVideoTimeUpdate = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    const time = event.currentTarget.currentTime;
    setPlaybackTime(time);

    const nextIndex = findActiveTranscriptIndex(time);
    const normalizedIndex = nextIndex >= 0 ? nextIndex : null;
    setActiveTranscriptIndex(prev => (prev === normalizedIndex ? prev : normalizedIndex));
  };

  const handleSeekVideo = (seconds: number) => {
    setActiveTab('playback');
    setPendingSeekTime(seconds);
    setPlaybackTime(seconds);
    const nextIndex = findActiveTranscriptIndex(seconds);
    setActiveTranscriptIndex(nextIndex >= 0 ? nextIndex : null);
  };

  useEffect(() => {
    if (activeTab !== 'playback' || pendingSeekTime === null || !videoRef.current) return;
    const seekTime = pendingSeekTime;
    if (videoRef.current) {
      videoRef.current.currentTime = seekTime;
      setPlaybackTime(seekTime);
      videoRef.current.play().catch(() => {});
    }
    setPendingSeekTime(null);
  }, [activeTab, pendingSeekTime]);

  useEffect(() => {
    if (activeTab !== 'playback' || activeTranscriptIndex === null) return;
    activeTranscriptRef.current?.scrollIntoView({
      block: 'center',
      behavior: 'smooth'
    });
  }, [activeTab, activeTranscriptIndex]);

  // --- CHAT ACTIONS ---
  const handleSendChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatQuestion.trim()) return;

    const userMsg = chatQuestion;
    setChatQuestion('');
    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        question: userMsg,
        model: llmModel
      });
      const engine = res.data.engine ? `\n\nEngine: ${res.data.engine}` : '';
      const sources = Array.isArray(res.data.sources) && res.data.sources.length
        ? `\nSources: ${res.data.sources.join(', ')}`
        : '';
      setChatHistory(prev => [...prev, { role: 'assistant', content: `${res.data.answer}${engine}${sources}` }]);
    } catch (err: any) {
      const errMsg = err.response?.data?.detail || err.message || 'Error communicating with AI.';
      setChatHistory(prev => [...prev, { role: 'assistant', content: `Chat Grounding Error: ${errMsg}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Formatter helper
  const formatSecs = (seconds: number) => {
    const s = Math.max(0, Math.floor(seconds || 0));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m.toString().padStart(2, '0')}:${rem.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`app-container ${sidebarCollapsed ? 'sidebar-collapsed' : ''} ${focusMode ? 'focus-mode' : ''}`}>
      {/* 1. Left Side Control Panel */}
      <aside className="sidebar">
        <div className="sidebar-title-container">
          <div className="sidebar-logo">
            <Cpu className="logo-icon" size={28} />
            <h1 className="logo-text">EchoNotes AI</h1>
          </div>
          <span className="subtitle">Multimodal NLP Lecture Studio</span>
        </div>

        <div className="glass-panel" style={{ padding: '1rem', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'center', marginBottom: '0.75rem' }}>
            <span className="input-label" style={{ marginBottom: 0 }}>Lecture Library</span>
            <span style={{
              fontSize: '0.72rem',
              color: systemStatus.database?.connected ? 'var(--green-complete)' : 'var(--red-accent)',
              fontWeight: 700
            }}>
              {systemStatus.database?.connected ? 'PostgreSQL online' : 'Database offline'}
            </span>
          </div>

          <select
            className="form-select"
            value={selectedProjectId}
            onChange={e => setSelectedProjectId(e.target.value)}
            disabled={!projects.length}
          >
            {projects.length === 0 ? (
              <option value="">No saved projects yet</option>
            ) : (
              projects.map(project => (
                <option key={project.id} value={project.id}>
                  {project.title}
                </option>
              ))
            )}
          </select>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.75rem' }}>
            <button className="btn-secondary" onClick={handleLoadProject} disabled={!selectedProjectId}>
              Load
            </button>
            <button className="btn-secondary" onClick={handleSaveCurrentProject} disabled={!systemStatus.active_project_id && !systemStatus.active_video_name}>
              Save
            </button>
          </div>
        </div>

        {/* Data Source Selector */}
        <div className="input-group">
          <label className="input-label">Data Source Entry Mode</label>
          <div className="radio-switch">
            <button 
              className={`radio-switch-btn ${sourceMode === 'upload' ? 'active' : ''}`}
              onClick={() => setSourceMode('upload')}
            >
              Upload
            </button>
            <button 
              className={`radio-switch-btn ${sourceMode === 'teams' ? 'active' : ''}`}
              onClick={() => setSourceMode('teams')}
            >
              Teams Link
            </button>
            <button 
              className={`radio-switch-btn ${sourceMode === 'demo' ? 'active' : ''}`}
              onClick={() => setSourceMode('demo')}
            >
              Demo Mock
            </button>
          </div>
        </div>

        {/* Dynamic Source Inputs */}
        {sourceMode === 'upload' && (
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '1.25rem' }}>
            <div className="input-group">
              <span className="input-label">Lecture Video (MP4/MKV)</span>
              <div className="file-upload-wrapper">
                <input type="file" className="file-upload-input" accept="video/*" onChange={handleVideoUpload} />
                <div className="file-upload-button">
                  <Upload size={16} /> {videoFile ? videoFile.name : 'Select video file'}
                </div>
              </div>
            </div>

            <div className="input-group">
              <span className="input-label">Teams Transcript (Optional SRT/VTT)</span>
              <div className="file-upload-wrapper">
                <input type="file" className="file-upload-input" accept=".vtt,.srt,.txt" onChange={handleTranscriptUpload} />
                <div className="file-upload-button">
                  <FileText size={16} /> {transcriptFile ? transcriptFile.name : 'Select transcript file'}
                </div>
              </div>
            </div>
            {uploadStatusText && <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textAlign: 'center' }}>{uploadStatusText}</p>}
          </div>
        )}

        {sourceMode === 'teams' && (
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '1.25rem' }}>
            <div className="input-group">
              <label className="input-label">SharePoint/Teams Recording URL</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="https://...sharepoint.com/..." 
                value={teamsUrl}
                onChange={e => setTeamsUrl(e.target.value)}
              />
            </div>
            
            {!teamsTokenReady && !deviceFlow && (
              <button className="btn-secondary" style={{ width: '100%', marginBottom: '0.5rem' }} onClick={handleMicrosoftSignIn}>
                Create Device sign-in code
              </button>
            )}

            {deviceFlow && (
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.75rem', borderRadius: '8px', marginBottom: '0.75rem' }}>
                <p style={{ fontSize: '0.8rem', marginBottom: '0.5rem' }}>Enter the user code at Microsoft Login:</p>
                <code style={{ fontSize: '1.1rem', color: 'var(--amber-status)', display: 'block', textAlign: 'center', margin: '0.5rem 0', fontWeight: 'bold' }}>
                  {deviceFlow.user_code}
                </code>
                <a href={deviceFlow.verification_uri} target="_blank" rel="noreferrer" style={{ color: 'var(--blue-normal)', fontSize: '0.8rem', textDecoration: 'underline', display: 'block', textAlign: 'center', marginBottom: '0.5rem' }}>
                  Open Microsoft Login Page
                </a>
                <button className="btn-primary" onClick={handleCompleteSignIn}>
                  Verify Sign-in Complete
                </button>
              </div>
            )}

            {teamsTokenReady && (
              <p style={{ color: 'var(--green-complete)', fontSize: '0.85rem', marginBottom: '0.75rem', textAlign: 'center' }}>
                Authenticated with Microsoft Graph.
              </p>
            )}

            <button 
              className="btn-primary" 
              onClick={handleTeamsDownload} 
              disabled={!teamsTokenReady || !teamsUrl}
            >
              Download Recording
            </button>
          </div>
        )}

        {sourceMode === 'demo' && (
          <div className="glass-panel" style={{ padding: '1rem', marginBottom: '1.25rem', textAlign: 'center' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Load a sample pre-processed Transformer & Attention lecture dataset instantly.
            </p>
            <button className="btn-primary" onClick={handleDemoLaunch}>
              Load Demo Dataset
            </button>
          </div>
        )}

        {/* AI Engine Settings */}
        <span className="sidebar-section-title"><Sparkles size={14} /> Speech Recognition</span>
        
        <div className="input-group">
          <label className="input-label">Whisper Model Size</label>
          <select className="form-select" value={whisperModel} onChange={e => setWhisperModel(e.target.value)}>
            <option value="tiny">Tiny (Fastest)</option>
            <option value="base">Base</option>
            <option value="small">Small (Default)</option>
            <option value="medium">Medium</option>
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">Local AI Model (Ollama)</label>
          <select className="form-select" value={llmModel} onChange={e => setLlmModel(e.target.value)}>
            {(ollamaModels.length > 0 ? ollamaModels : [llmModel]).map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">Vision Model (Ollama VLM)</label>
          <select className="form-select" value={visionModel} onChange={e => setVisionModel(e.target.value)}>
            {Array.from(new Set([visionModel, ...ollamaModels, 'llava:7b'])).map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">Spoken Language</label>
          <select className="form-select" value={spokenLanguage} onChange={e => setSpokenLanguage(e.target.value)}>
            <option value="en">English (Recommended)</option>
            <option value="vi">Vietnamese (Forced)</option>
            <option value="auto">Auto Detect (can misread accents)</option>
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">ASR Context Prompt</label>
          <textarea 
            className="form-textarea" 
            value={asrPrompt} 
            onChange={e => setAsrPrompt(e.target.value)}
          />
        </div>

        <div className="input-group">
          <label className="input-label">ASR Hotwords List</label>
          <input 
            type="text" 
            className="form-input" 
            placeholder="inode allocation partition" 
            value={hotwords}
            onChange={e => setHotwords(e.target.value)}
          />
        </div>

        {/* Vision and Acoustic Settings */}
        <span className="sidebar-section-title"><Eye size={14} /> Visual & Acoustic Analysis</span>

        <div className="input-group">
          <label className="input-label">Visual Mode</label>
          <select className="form-select" value={visualMode} onChange={e => setVisualMode(e.target.value)}>
            <option value="Transcript only: skip slides/keyframes">Transcript only</option>
            <option value="Fast: capture keyframes, no OCR">Fast Keyframes (No OCR)</option>
            <option value="Full: capture keyframes + OCR">Full Keyframes + OCR</option>
            <option value="VLM: keyframes + image understanding">VLM Keyframes + Image Understanding</option>
          </select>
        </div>

        {visualMode !== 'Transcript only: skip slides/keyframes' && (
          <>
            <div className="slider-container">
              <div className="slider-header">
                <span>SSIM Slide Sensitivity</span>
                <span>{ssimSensitivity}</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min={0.85} 
                max={0.99} 
                step={0.01} 
                value={ssimSensitivity} 
                onChange={e => setSsimSensitivity(parseFloat(e.target.value))}
              />
            </div>

            <div className="control-grid-2">
              <div className="input-group">
                <label className="input-label" style={{ fontSize: '0.75rem' }}>Min Slide Gap (s)</label>
                <input 
                  type="number" 
                  className="form-input" 
                  value={minKeyframeGap} 
                  onChange={e => setMinKeyframeGap(parseInt(e.target.value) || 10)}
                />
              </div>
              <div className="input-group">
                <label className="input-label" style={{ fontSize: '0.75rem' }}>Max Slides</label>
                <input 
                  type="number" 
                  className="form-input" 
                  value={maxKeyframes} 
                  onChange={e => setMaxKeyframes(parseInt(e.target.value) || 20)}
                />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Frame Scan Interval (s)</label>
              <input 
                type="number" 
                className="form-input" 
                value={frameCheckInterval} 
                onChange={e => setFrameCheckInterval(parseInt(e.target.value) || 5)}
              />
            </div>
          </>
        )}

        <label className="checkbox-label">
          <input 
            type="checkbox" 
            className="checkbox-input" 
            checked={acousticToggle}
            onChange={e => setAcousticToggle(e.target.checked)}
          />
          Enable Acoustic Emphasis
        </label>

        <label className="checkbox-label">
          <input 
            type="checkbox" 
            className="checkbox-input" 
            checked={useOsGlossary}
            onChange={e => setUseOsGlossary(e.target.checked)}
          />
          OS Technical Glossary Rules
        </label>

        {/* Global actions */}
        <div style={{ marginTop: 'auto', paddingTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <button 
            className="btn-primary" 
            onClick={handleRunAnalysis}
            disabled={systemStatus.status === 'processing' || (!systemStatus.active_video_name && !videoFile)}
          >
            <Play size={16} /> Run Pipeline
          </button>
          
          <button className="btn-danger" style={{ width: '100%' }} onClick={handleClearCache}>
            Start New Lecture Session
          </button>
        </div>
      </aside>

      {/* 2. Main Workspace Layout */}
      <main className="main-content">
        <div className="workspace-controls">
          <button className="workspace-toggle-btn" onClick={() => setSidebarCollapsed(prev => !prev)}>
            {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
            {sidebarCollapsed ? 'Show Controls' : 'Hide Controls'}
          </button>
          <button className="workspace-toggle-btn" onClick={() => setFocusMode(prev => !prev)}>
            {focusMode ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            {focusMode ? 'Exit Focus' : 'Focus Workspace'}
          </button>
        </div>

        <header className="main-header">
          <div className="header-title-area">
            <h2>EchoNotes AI Dashboard</h2>
            <p>
              {systemStatus.active_video_name 
                ? `Active Lecture Analysis: ${systemStatus.active_video_name}` 
                : 'Upload or Load a Lecture dataset to run multimodal synthesizers.'
              }
            </p>
          </div>

        </header>

        {/* 3. Signal Processing Timeline */}
        <section className="timeline-section">
          <div className="timeline-container">
            <div className={`timeline-step ${systemStatus.status === 'completed' || systemStatus.progress > 10 ? 'completed' : ''} ${systemStatus.status === 'processing' && systemStatus.progress <= 15 ? 'active' : ''}`}>
              <span className="step-label">Step 1/5 {systemStatus.progress <= 15 && systemStatus.status === 'processing' && '- Running'}</span>
              <span className="step-name">Input Prepared</span>
            </div>

            <div className={`timeline-step ${systemStatus.status === 'completed' || systemStatus.progress > 30 ? 'completed' : ''} ${systemStatus.status === 'processing' && systemStatus.progress > 15 && systemStatus.progress <= 55 ? 'active' : ''}`}>
              <span className="step-label">Step 2/5 {systemStatus.progress > 15 && systemStatus.progress <= 55 && '- Running'}</span>
              <span className="step-name">Audio & Speech ASR</span>
            </div>

            <div className={`timeline-step ${systemStatus.status === 'completed' || systemStatus.progress > 65 ? 'completed' : ''} ${systemStatus.status === 'processing' && systemStatus.progress > 55 && systemStatus.progress <= 75 ? 'active' : ''}`}>
              <span className="step-label">Step 3/5 {systemStatus.progress > 55 && systemStatus.progress <= 75 && '- Running'}</span>
              <span className="step-name">Acoustic Labels</span>
            </div>

            <div className={`timeline-step ${systemStatus.status === 'completed' || systemStatus.progress > 85 ? 'completed' : ''} ${systemStatus.status === 'processing' && systemStatus.progress > 75 && systemStatus.progress <= 90 ? 'active' : ''}`}>
              <span className="step-label">Step 4/5 {systemStatus.progress > 75 && systemStatus.progress <= 90 && '- Running'}</span>
              <span className="step-name">Visual Keyframes</span>
            </div>

            <div className={`timeline-step ${systemStatus.status === 'completed' ? 'completed' : ''} ${systemStatus.status === 'processing' && systemStatus.progress > 90 ? 'active' : ''}`}>
              <span className="step-label">Step 5/5 {systemStatus.progress > 90 && systemStatus.status === 'processing' && '- Finishing'}</span>
              <span className="step-name">NLP Report Sync</span>
            </div>
          </div>

          {systemStatus.status === 'processing' && (
            <div className="workstation-loading" style={{ marginTop: '0.75rem' }}>
              <span className="workstation-loading-text">
                <RefreshCw size={14} className="pulse" /> Status: {systemStatus.stage}...
              </span>
              <div className="workstation-loading-bar">
                <div className="workstation-loading-progress" style={{ width: `${systemStatus.progress}%` }} />
              </div>
            </div>
          )}
        </section>

        {/* 4. Multimodal Workstation Area */}
        <div className="workspace-grid">
          {/* Tabs Navigation */}
          <nav className="tabs-nav">
            <button className={`tab-btn ${activeTab === 'notes' ? 'active' : ''}`} onClick={() => setActiveTab('notes')}>
              NLP Smart Notes
            </button>
            <button className={`tab-btn ${activeTab === 'playback' ? 'active' : ''}`} onClick={() => setActiveTab('playback')}>
              Lecture Playback
            </button>
            <button className={`tab-btn ${activeTab === 'visual' ? 'active' : ''}`} onClick={() => setActiveTab('visual')}>
              Visual Context
            </button>
            <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
              AI Grounded Q&A
            </button>
          </nav>

          {/* Tabs Window */}
          <div className="tab-window">
            
            {/* TAB: SMART NOTES */}
            {activeTab === 'notes' && (
              <div className="report-view-container">
                <div className="report-controls">
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <label className="checkbox-label" style={{ marginBottom: 0 }}>
                      <input 
                        type="radio" 
                        name="notes_mode"
                        checked={synthesisMethod === 'fast'}
                        onChange={() => setSynthesisMethod('fast')} 
                      /> Fast Synthesis
                    </label>
                    <label className="checkbox-label" style={{ marginBottom: 0 }}>
                      <input 
                        type="radio" 
                        name="notes_mode"
                        checked={synthesisMethod === 'ollama'}
                        onChange={() => setSynthesisMethod('ollama')} 
                      /> Local AI Synthesis (Ollama)
                    </label>
                  </div>

                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    {systemStatus.generating_report ? (
                      <>
                        <button className="btn-secondary" disabled>
                          <RefreshCw size={14} className="pulse" /> Synthesizing notes...
                        </button>
                        <button className="btn-secondary" onClick={handleResetReportGeneration} title="Reset a stuck local synthesis run">
                          Reset
                        </button>
                      </>
                    ) : (
                      <button className="btn-primary" style={{ padding: '0.5rem 1rem' }} onClick={handleGenerateReport}>
                        Generate Notes
                      </button>
                    )}

                    {reportMarkdown && (
                      <>
                        <button className="btn-secondary" onClick={handleDownloadMarkdown} title="Download Markdown">
                          <Download size={14} /> Markdown
                        </button>
                        <button className="btn-secondary" onClick={handleDownloadPDF} title="Export PDF">
                          <FileDown size={14} /> PDF
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {topicBlocks.length > 0 && (
                  <section className="semantic-topic-panel">
                    <div className="semantic-topic-header">
                      <div>
                        <h3>Semantic Topic Map</h3>
                        <p>Local NLP segmentation used by the hybrid RAG engine.</p>
                      </div>
                      <span>{topicBlocks.length} topics</span>
                    </div>
                    <div className="semantic-topic-grid">
                      {topicBlocks.slice(0, 10).map((block, idx) => (
                        <button
                          key={`${block.start}-${idx}`}
                          className="semantic-topic-card"
                          onClick={() => handleSeekVideo(block.start)}
                          title="Jump playback to this topic"
                        >
                          <span className="semantic-topic-time">
                            {block.timestamp} - {block.end_timestamp}
                          </span>
                          <strong>{block.title || `Topic ${idx + 1}`}</strong>
                          <span className="semantic-topic-meta">
                            {block.segment_count || 0} segments · {block.word_count || 0} words
                          </span>
                          <span className="semantic-topic-keywords">
                            {(block.keywords || []).slice(0, 5).map((kw: string) => `#${kw}`).join(' ')}
                          </span>
                        </button>
                      ))}
                    </div>
                  </section>
                )}

                <div className="glass-panel report-editor-card" style={{ flex: 1, minHeight: '350px' }}>
                  <div style={{ overflowY: 'auto', height: '100%' }}>
                    <SimpleMarkdown markdown={reportMarkdown} />
                  </div>
                </div>
              </div>
            )}

            {/* TAB: LECTURE PLAYBACK */}
            {activeTab === 'playback' && (
              <div className="playback-layout">
                {/* Left: Video Player */}
                <div className="video-container">
                  <div className="glass-panel" style={{ padding: '0.5rem', borderRadius: '12px' }}>
                    <video 
                      ref={videoRef} 
                      src={`${API_BASE}/video`} 
                      controls 
                      className="html5-player"
                      onTimeUpdate={handleVideoTimeUpdate}
                      onSeeked={handleVideoTimeUpdate}
                    />
                  </div>
                  <div className="input-group" style={{ marginTop: '0.5rem' }}>
                    <div style={{ position: 'relative' }}>
                      <Search size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
                      <input 
                        type="text" 
                        className="search-transcript-input"
                        style={{ paddingLeft: '2.5rem' }}
                        placeholder="Search transcript text..." 
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                {/* Right: Timestamped Scrollable Transcript */}
                <div className="transcript-panel">
                  <h3 style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    Acoustic & Speech Timeline Navigator
                    <span className="live-playback-pill">Now {formatSecs(playbackTime)}</span>
                  </h3>
                  
                  {transcript.length === 0 ? (
                    <div className="empty-state-panel">
                      <Video className="empty-state-icon" size={32} />
                      <p>Run lecture pipeline to generate timeline transcripts.</p>
                    </div>
                  ) : (
                    <div className="transcript-scroll">
                      {transcript
                        .map((seg, idx) => ({ seg, idx }))
                        .filter(({ seg }) => !searchQuery || seg.text.toLowerCase().includes(searchQuery.toLowerCase()))
                        .map(({ seg, idx }) => {
                          const isImportant = seg.acoustics?.is_important;
                          const isActive = idx === activeTranscriptIndex;
                          return (
                            <div 
                              key={idx} 
                              ref={isActive ? activeTranscriptRef : undefined}
                              className={`transcript-row ${isImportant ? 'red-row' : 'blue-row'} ${isActive ? 'active-row' : ''}`}
                              onClick={() => handleSeekVideo(seg.start)}
                              aria-current={isActive ? 'true' : undefined}
                            >
                              <div className="row-timestamp">
                                {formatSecs(seg.start)}
                              </div>
                              <div className="row-text">
                                {seg.text}
                              </div>
                            </div>
                          );
                        })
                      }
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB: VISUAL CONTEXT */}
            {activeTab === 'visual' && (
              <div className="view-container">
                {slides.length === 0 ? (
                  <div className="empty-state-panel">
                    <Eye className="empty-state-icon" size={40} />
                    <h3>No Slide Context captured</h3>
                    <p>Make sure slide detection is enabled and ASR run is fully completed.</p>
                  </div>
                ) : (
                  <div className="keyframes-grid">
                    {slides.map((slide, idx) => (
                      <div key={idx} className="glass-panel keyframe-card">
                        <div className="keyframe-header">
                          <span className="keyframe-timestamp">Slide capture @ {slide.timestamp_formatted}</span>
                          <button className="btn-secondary" style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }} onClick={() => handleSeekVideo(slide.timestamp_sec)}>
                            Jump Video
                          </button>
                        </div>
                        <div className="keyframe-img-box">
                          {slide.image_path ? (
                            <img 
                              src={`${API_BASE}/keyframes/${slide.image_path}`} 
                              alt={`Keyframe ${idx}`}
                              className="keyframe-img"
                              onError={(e) => { e.currentTarget.style.display = 'none'; }}
                            />
                          ) : (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                              No Image Preview
                            </div>
                          )}
                        </div>
                        <div className="keyframe-info">
                          {slide.vlm_description && (
                            <div className="keyframe-ocr-box">
                              <strong>VLM image understanding:</strong>
                              <p style={{ marginTop: '0.25rem', whiteSpace: 'pre-wrap' }}>{slide.vlm_description}</p>
                            </div>
                          )}
                          {slide.ocr_text && (
                            <div className="keyframe-ocr-box">
                              <strong>Extracted slide text:</strong>
                              <p style={{ marginTop: '0.25rem', whiteSpace: 'pre-wrap' }}>{slide.ocr_text}</p>
                            </div>
                          )}
                          <div className="keyframe-transcript-nearby">
                            <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 'bold' }}>Aligned spoken text:</span>
                            <p style={{ marginTop: '0.25rem', fontStyle: 'italic' }}>
                              {(() => {
                                const nextSlide = slides[idx + 1];
                                const end = nextSlide ? nextSlide.timestamp_sec : 999999;
                                const related = transcript
                                  .filter(t => t.start >= slide.timestamp_sec && t.start < end)
                                  .map(t => t.text)
                                  .join(' ');
                                return related.substring(0, 180) + (related.length > 180 ? '...' : '') || 'No speech segments aligned to this slide.';
                              })()}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* TAB: AI CHAT */}
            {activeTab === 'chat' && (
              <div className="chat-tab-container">
                <div className="glass-panel chat-messages-box">
                  {chatHistory.length === 0 && (
                    <div className="empty-state-panel" style={{ height: '100%' }}>
                      <MessageSquare className="empty-state-icon" size={40} />
                      <h3>AI Grounded Q&A Assistant</h3>
                      <p>Ask queries specifically grounded to the lecture report, transcript context, and acoustic emphasis signals.</p>
                    </div>
                  )}

                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`chat-bubble-wrapper ${msg.role === 'user' ? 'user' : 'assistant'}`}>
                      <div className="chat-bubble-avatar">
                        {msg.role === 'user' ? 'U' : 'AI'}
                      </div>
                      <div>
                        <div className="chat-bubble-meta">
                          {msg.role === 'user' ? 'You' : 'EchoNotes Assistant'}
                        </div>
                        <div className="chat-bubble-content">
                          {msg.content}
                        </div>
                      </div>
                    </div>
                  ))}

                  {chatLoading && (
                    <div className="chat-bubble-wrapper assistant">
                      <div className="chat-bubble-avatar">AI</div>
                      <div style={{ width: '100%' }}>
                        <div className="chat-bubble-meta">EchoNotes Assistant</div>
                        <div className="chat-bubble-content" style={{ width: '100%' }}>
                          <div className="workstation-loading" style={{ padding: '0.5rem', background: 'transparent', border: 'none', boxShadow: 'none' }}>
                            <span className="workstation-loading-text" style={{ fontSize: '0.8rem' }}>
                              Running LangChain RAG over report and transcript...
                            </span>
                            <div className="workstation-loading-bar" style={{ marginTop: '0.35rem' }}>
                              <div className="workstation-loading-progress pulse" style={{ width: '80%' }} />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <form className="chat-input-panel" onSubmit={handleSendChat}>
                  <input 
                    type="text" 
                    className="chat-text-input" 
                    placeholder="Ask grounded questions about terms, slides, or teacher timestamps..." 
                    value={chatQuestion}
                    onChange={e => setChatQuestion(e.target.value)}
                    disabled={chatLoading}
                  />
                  <button className="chat-send-btn" type="submit" disabled={chatLoading || !chatQuestion.trim()}>
                    Send Query
                  </button>
                </form>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  );
}
