import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, Video, FileText, Cpu, Eye, MessageSquare,
  RefreshCw, Upload, Sparkles, Download, FileDown, Search,
  PanelLeftClose, PanelLeftOpen, Maximize2, Minimize2, ChevronDown
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000/api';
type Citation = { label: string; source?: string; title?: string; timestamp?: string; start?: number | null; snippet?: string };
type ChatItem = { role: 'user' | 'assistant'; content: string; citations?: Citation[]; engine?: string; message_id?: string; latency_ms?: number; mode?: string };

const CHAT_MODE_OPTIONS = [
  {
    value: 'explain',
    label: 'Explain concept',
    description: 'Grounded explanation from transcript, report, and visual context.'
  },
  {
    value: 'find_moment',
    label: 'Find exact moment',
    description: 'Return the most relevant timestamps and evidence snippets.'
  },
  {
    value: 'quiz',
    label: 'Generate quiz',
    description: 'Create review questions from the current lecture context.'
  },
  {
    value: 'summarize_range',
    label: 'Summarize range',
    description: 'Condense the matching time window into concise notes.'
  },
  {
    value: 'compare_visual',
    label: 'Compare slide vs transcript',
    description: 'Align slide/keyframe evidence with what was spoken.'
  }
];

function formatAblationMode(mode: string) {
  const labels: Record<string, string> = {
    transcript_only: 'Transcript only',
    ocr_only: 'OCR only',
    vlm_only: 'Image understanding only',
    ocr_plus_vlm_plus_transcript: 'OCR + image understanding + transcript'
  };
  return labels[mode] || mode.replace(/_/g, ' ');
}

function formatMetricValue(value: unknown) {
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)));
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    if (record.status || record.provider) {
      return [record.provider, record.status].filter(Boolean).join(' / ');
    }
    return JSON.stringify(value);
  }
  return String(value);
}

function formatSpeakerLabel(seg: any) {
  const label = seg?.speaker || seg?.speaker_id || 'unknown';
  if (label === 'unknown') {
    const role = formatSpeakerRole(seg);
    return role === 'participant' ? 'unlabeled' : `${role} cue`;
  }
  return String(label).replace(/^SPEAKER_/, 'SPK ');
}

function formatSpeakerRole(seg: any) {
  return String(seg?.speaker_role || seg?.role || 'participant').toLowerCase();
}

function cleanMarkdownText(text: string) {
  return (text || "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();
}

function renderInlineMarkdown(text: string) {
  const parts = (text || "").split(/(\*\*.*?\*\*|__.*?__|`.*?`|\*.*?\*|_[^_]+_)/g).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={idx} className="inline-markdown-bold">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("__") && part.endsWith("__")) {
      return <strong key={idx} className="inline-markdown-bold">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={idx} className="inline-markdown-code">{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={idx} className="inline-markdown-italic" style={{ fontStyle: 'italic' }}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("_") && part.endsWith("_")) {
      return <em key={idx} className="inline-markdown-italic" style={{ fontStyle: 'italic' }}>{part.slice(1, -1)}</em>;
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function parseQuizItems(content: string) {
  const normalized = (content || "")
    .replace(/\r/g, "")
    .replace(/\s+-\s+\*\*Answer Hint:\*\*/g, "\nAnswer Hint:")
    .replace(/\s+-\s+Answer Hint:/g, "\nAnswer Hint:")
    .replace(/\s*(\d+)\.\s+\*\*Question:\*\*/g, "\n$1. Question:")
    .replace(/\s*(\d+)\.\s+Question:/g, "\n$1. Question:")
    .replace(/\*\*/g, "");
  const matches = [...normalized.matchAll(/(?:^|\n)\s*(\d+)\.\s*Question:\s*([\s\S]*?)(?=\n\s*\d+\.\s*Question:|$)/g)];
  return matches.map(match => {
    const raw = match[2].trim();
    const [questionPart, answerPart = ""] = raw.split(/\n\s*Answer Hint:\s*/);
    return {
      index: Number(match[1]),
      question: cleanMarkdownText(questionPart),
      hint: cleanMarkdownText(answerPart),
    };
  }).filter(item => item.question);
}

function ChatMessageContent({ content, mode }: { content: string; mode?: string }) {
  const looksLikeQuiz = /\bQuestion:\b|\*\*Question:\*\*/i.test(content || "");
  const quizItems = mode === "quiz" || looksLikeQuiz ? parseQuizItems(content) : [];
  if (quizItems.length > 0) {
    return (
      <div className="quiz-answer-list">
        <div className="quiz-answer-kicker">Generated Quiz</div>
        {quizItems.map(item => (
          <div className="quiz-answer-card" key={item.index}>
            <div className="quiz-answer-index">{item.index}</div>
            <div className="quiz-answer-body">
              <strong>{item.question}</strong>
              {item.hint && <p>{item.hint}</p>}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="chat-markdown">
      {(content || "").split(/\n{2,}/).filter(Boolean).map((paragraph, idx) => (
        <p key={idx}>{renderInlineMarkdown(paragraph.trim())}</p>
      ))}
    </div>
  );
}

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
    // Handles trailing spaces or Windows carriage returns (\r) robustly.
    const imgRegex = /^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$/;
    const imgMatch = line.match(imgRegex);
    if (imgMatch) {
      let imageSrc = imgMatch[2].trim();
      if (!imageSrc.startsWith('http') && !imageSrc.startsWith('data:')) {
        const parts = imageSrc.split(/[/\\]/);
        const filename = parts[parts.length - 1];
        imageSrc = `${API_BASE}/keyframes/${filename}`;
      }
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
      elements.push(<h1 key={index}>{renderInlineMarkdown(line.substring(2))}</h1>);
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={index}>{renderInlineMarkdown(line.substring(3))}</h2>);
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={index}>{renderInlineMarkdown(line.substring(4))}</h3>);
    } else if (line.startsWith('#### ')) {
      elements.push(<h4 key={index}>{renderInlineMarkdown(line.substring(5))}</h4>);
    }
    // Blockquote
    else if (line.startsWith('> ')) {
      elements.push(<blockquote key={index} style={{ borderLeft: '3px solid var(--blue-normal)', paddingLeft: '1rem', color: 'var(--text-secondary)', margin: '1rem 0', fontStyle: 'italic' }}>{renderInlineMarkdown(line.substring(2))}</blockquote>);
    }
    // Bullets
    else if (line.trim().startsWith('- ') || line.trim().startsWith('* ') || line.trim().startsWith('+ ')) {
      elements.push(<li key={index} style={{ marginLeft: '1.5rem', marginTop: '0.25rem', marginBottom: '0.25rem' }}>{renderInlineMarkdown(line.trim().substring(2))}</li>);
    }
    // Divider
    else if (line.trim() === '---') {
      elements.push(<hr key={index} style={{ border: 'none', borderBottom: '1px solid var(--border-color)', margin: '1.5rem 0' }} />);
    }
    // Paragraph / Normal line
    else if (line.trim()) {
      elements.push(<p key={index} style={{ marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>{renderInlineMarkdown(line)}</p>);
    }
  });

  return <div className="report-markdown-preview">{elements}</div>;
}

export default function App() {
  // Sidebar states
  const [sourceMode, setSourceMode] = useState<'upload' | 'teams' | 'demo'>('upload');
  const [whisperModel, setWhisperModel] = useState('small');
  const [llmModel, setLlmModel] = useState('qwen2.5:1.5b-instruct');
  const [visionModel, setVisionModel] = useState('llava:7b');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [spokenLanguage, setSpokenLanguage] = useState('auto');
  const [asrPrompt, setAsrPrompt] = useState('Technical lecture. Preserve the speaker\'s original language. Keep technical terms and code identifiers unchanged. Do not translate.');
  const [hotwords, setHotwords] = useState('');
  const [visualMode, setVisualMode] = useState('Fast: capture keyframes, no OCR');
  const [ssimSensitivity, setSsimSensitivity] = useState(0.94);
  const [minKeyframeGap, setMinKeyframeGap] = useState(20);
  const [maxKeyframes, setMaxKeyframes] = useState(80);
  const [frameCheckInterval, setFrameCheckInterval] = useState(10);
  const [acousticToggle, setAcousticToggle] = useState(false);
  const [diarizationToggle, setDiarizationToggle] = useState(false);
  const [useOsGlossary, setUseOsGlossary] = useState(true);

  const languagePrompts: Record<string, string> = {
    auto: 'Technical lecture. Preserve the speaker\'s original language. Keep technical terms and code identifiers unchanged. Do not translate.',
    en: 'English technical lecture. Preserve exact English terms, code identifiers, product names, and acronyms. Do not translate.',
    vi: 'Bài giảng tiếng Việt có thể xen thuật ngữ tiếng Anh. Nhận diện và giữ nguyên tiếng Việt; giữ nguyên thuật ngữ tiếng Anh/code identifier nếu người nói dùng. Không dịch sang tiếng Anh.'
  };

  const handleSpokenLanguageChange = (language: string) => {
    setSpokenLanguage(language);
    setAsrPrompt(languagePrompts[language] || languagePrompts.auto);
    const staleHotwords = ['inode allocation partition', 'inode', 'allocation', 'partition'];
    if (language === 'vi' && staleHotwords.some(term => hotwords.toLowerCase().includes(term))) {
      setHotwords('');
    }
  };

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
  const [projectSearch, setProjectSearch] = useState('');
  const [projectCourse, setProjectCourse] = useState('');
  const [projectTags, setProjectTags] = useState('');
  const [projectDescription, setProjectDescription] = useState('');

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
    database: { connected: false, error: null as string | null, provider: '' as string, fallback_reason: null as string | null }
  });

  // Multimodal outputs
  const [activeTab, setActiveTab] = useState<'notes' | 'visual' | 'playback' | 'chat' | 'evaluation'>('notes');
  const [transcript, setTranscript] = useState<any[]>([]);
  const [slides, setSlides] = useState<any[]>([]);
  const [topicBlocks, setTopicBlocks] = useState<any[]>([]);
  const [reportMarkdown, setReportMarkdown] = useState('');
  const [synthesisMethod, setSynthesisMethod] = useState<'fast' | 'ollama'>('fast');
  const [searchQuery, setSearchQuery] = useState('');
  const [playbackTime, setPlaybackTime] = useState(0);
  const [activeTranscriptIndex, setActiveTranscriptIndex] = useState<number | null>(null);
  const [pendingSeekTime, setPendingSeekTime] = useState<number | null>(null);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [focusMode, setFocusMode] = useState(false);

  // Grounded Chat state
  const [chatHistory, setChatHistory] = useState<ChatItem[]>([]);
  const [chatQuestion, setChatQuestion] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatMode, setChatMode] = useState('explain');
  const [chatModeOpen, setChatModeOpen] = useState(false);
  const [evaluationSummary, setEvaluationSummary] = useState<any | null>(null);
  const [speakerMapping, setSpeakerMapping] = useState<Record<string, string>>({});

  // Ref handles
  const videoRef = useRef<HTMLVideoElement>(null);
  const activeTranscriptRef = useRef<HTMLDivElement | null>(null);
  const chatModeRef = useRef<HTMLDivElement | null>(null);

  // Fetch initial Ollama models list & status
  useEffect(() => {
    axios.get(`${API_BASE}/ollama-models`)
      .then(res => {
        if (res.data.models && res.data.models.length > 0) {
          setOllamaModels(res.data.models);
          // Set default to qwen 1.5B if available (safer for RAM/VRAM), then 7B, or first model
          if (res.data.models.includes('qwen2.5:1.5b-instruct')) {
            setLlmModel('qwen2.5:1.5b-instruct');
          } else if (res.data.models.includes('qwen2.5:7b-instruct')) {
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

  useEffect(() => {
    const project = projects.find(p => p.id === selectedProjectId);
    if (!project) return;
    setProjectCourse(project.course_name || '');
    setProjectTags((project.tags || []).join(', '));
    setProjectDescription(project.description || '');
  }, [selectedProjectId, projects]);

  useEffect(() => {
    if (evaluationSummary?.speaker_roles?.speakers) {
      const mapping: Record<string, string> = {};
      evaluationSummary.speaker_roles.speakers.forEach((s: any) => {
        mapping[s.speaker] = s.speaker;
      });
      setSpeakerMapping(mapping);
    }
  }, [evaluationSummary]);

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
      const res = await axios.get(`${API_BASE}/projects`, { params: { search: projectSearch } });
      setProjects(res.data.projects || []);
      if (!selectedProjectId && res.data.projects?.length) {
        setSelectedProjectId(res.data.projects[0].id);
      }
    } catch {
      setProjects([]);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => fetchProjects(), 350);
    return () => clearTimeout(timer);
  }, [projectSearch]);

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
        analyze_acoustics_enabled: acousticToggle,
        diarization_enabled: diarizationToggle
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
      setSelectedProjectId('');
      setProjectCourse('');
      setProjectTags('');
      setProjectDescription('');
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
      setChatHistory((res.data.chat || []).map((m: any) => ({
        role: m.role,
        content: m.content,
        citations: m.citations || [],
        mode: m.query_mode,
        message_id: m.id,
        latency_ms: m.latency_ms
      })));
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
      alert('Current lecture project saved to the active project database.');
    } catch (err: any) {
      alert(`Save project failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleUpdateProjectMetadata = async () => {
    if (!selectedProjectId) return;
    try {
      await axios.post(`${API_BASE}/projects/${selectedProjectId}/metadata`, {
        course_name: projectCourse,
        tags: projectTags.split(',').map(tag => tag.trim()).filter(Boolean),
        description: projectDescription
      });
      fetchProjects();
    } catch (err: any) {
      alert(`Metadata save failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const fetchEvaluationSummary = async () => {
    try {
      const res = await axios.get(`${API_BASE}/evaluation/summary`);
      setEvaluationSummary(res.data);
    } catch {
      setEvaluationSummary(null);
    }
  };

  const handleIngestPdf = async () => {
    try {
      await axios.post(`${API_BASE}/report/ingest-pdf`);
      await fetchEvaluationSummary();
      alert('PDF artifact ingested into grounded Q&A context.');
    } catch (err: any) {
      alert(`PDF ingest failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleSyncStorage = async () => {
    try {
      const res = await axios.post(`${API_BASE}/storage/sync-current`);
      const count = Object.keys(res.data.artifacts || {}).length;
      alert(`Storage sync completed with ${count} artifacts using provider: ${res.data.provider}`);
    } catch (err: any) {
      alert(`Storage sync failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleSaveSpeakerMapping = async () => {
    if (!systemStatus.active_project_id) {
      alert('No active project to apply speaker mapping.');
      return;
    }
    try {
      await axios.post(`${API_BASE}/projects/${systemStatus.active_project_id}/speaker-map`, {
        speaker_map: speakerMapping
      });
      alert('Speaker mapping saved successfully!');
      fetchResults();
      fetchEvaluationSummary();
    } catch (err: any) {
      alert(`Save speaker mapping failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleReferenceTranscriptUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post(`${API_BASE}/upload/transcript`, formData);
      pollStatus();
      fetchEvaluationSummary();
    } catch (err: any) {
      alert(`Upload reference failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const handleBuildRegressionSet = async () => {
    try {
      const res = await axios.post(`${API_BASE}/evaluation/regression-set/build`);
      await fetchEvaluationSummary();
      alert(`Regression set saved with ${res.data.regression_set?.cases || 0} cases.`);
    } catch (err: any) {
      alert(`Regression set build failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  const downloadFromEndpoint = (endpoint: string) => {
    window.open(`${API_BASE}${endpoint}`, '_blank');
  };

  const handleChatFeedback = async (messageId: string | undefined, rating: string) => {
    if (!messageId) return;
    try {
      await axios.post(`${API_BASE}/chat/feedback`, { message_id: messageId, rating, comment: '' });
      setChatHistory(prev => prev.map(msg => msg.message_id === messageId ? { ...msg, content: `${msg.content}\n\nFeedback: ${rating}` } : msg));
    } catch {}
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
    if (typeof seconds !== 'number' || Number.isNaN(seconds)) return;
    setActiveTab('playback');
    const seekTime = Math.max(0, seconds);
    setPendingSeekTime(seekTime);
    setPlaybackTime(seekTime);
    const nextIndex = findActiveTranscriptIndex(seekTime);
    setActiveTranscriptIndex(nextIndex >= 0 ? nextIndex : null);
  };

  const activeVideoSrc = systemStatus.active_video_name
    ? `${API_BASE}/video?v=${encodeURIComponent(systemStatus.active_video_name)}`
    : '';

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

  useEffect(() => {
    if (activeTab === 'evaluation') {
      fetchEvaluationSummary();
    }
  }, [activeTab]);

  useEffect(() => {
    const closeChatMode = (event: MouseEvent) => {
      if (chatModeRef.current && !chatModeRef.current.contains(event.target as Node)) {
        setChatModeOpen(false);
      }
    };
    document.addEventListener('mousedown', closeChatMode);
    return () => document.removeEventListener('mousedown', closeChatMode);
  }, []);

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
        model: llmModel,
        mode: chatMode
      });
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: res.data.answer,
        citations: res.data.citations || [],
        engine: res.data.engine,
        mode: res.data.mode,
        latency_ms: res.data.latency_ms,
        message_id: res.data.message_id
      }]);
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

  const selectedChatMode = CHAT_MODE_OPTIONS.find(option => option.value === chatMode) || CHAT_MODE_OPTIONS[0];
  const databaseProvider = systemStatus.database?.provider || '';
  const databaseLabel = systemStatus.database?.connected
    ? (databaseProvider === 'sqlite-fallback' ? 'Local DB fallback' : 'PostgreSQL online')
    : 'Database offline';

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
              {databaseLabel}
            </span>
          </div>

          <input
            className="form-input"
            style={{ marginBottom: '0.6rem' }}
            placeholder="Filter by course, tag, title..."
            value={projectSearch}
            onChange={e => setProjectSearch(e.target.value)}
          />

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

          {selectedProjectId && (
            <div style={{ display: 'grid', gap: '0.5rem', marginTop: '0.75rem' }}>
              <input className="form-input" placeholder="Course name" value={projectCourse} onChange={e => setProjectCourse(e.target.value)} />
              <input className="form-input" placeholder="Tags: genai, bosch, training" value={projectTags} onChange={e => setProjectTags(e.target.value)} />
              <textarea className="form-textarea" rows={2} placeholder="Project description" value={projectDescription} onChange={e => setProjectDescription(e.target.value)} />
              <button className="btn-secondary" onClick={handleUpdateProjectMetadata}>Save Metadata</button>
            </div>
          )}
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
            <option value="large-v3">Large-v3 (Best quality, heavy)</option>
          </select>
          <p className="input-help">
            Large-v3 improves Vietnamese and mixed-language ASR but needs more VRAM/RAM and downloads on first use.
          </p>
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
          <label className="input-label">Image Understanding Model (Ollama)</label>
          <select className="form-select" value={visionModel} onChange={e => setVisionModel(e.target.value)}>
            {Array.from(new Set([visionModel, ...ollamaModels, 'llava:7b'])).map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label className="input-label">Spoken Language</label>
          <select className="form-select" value={spokenLanguage} onChange={e => handleSpokenLanguageChange(e.target.value)}>
            <option value="auto">Auto Detect (first-language guess)</option>
            <option value="vi">Vietnamese + English terms (mixed lecture)</option>
            <option value="en">English (force English ASR)</option>
          </select>
          <p className="input-help">
            For Vietnamese lectures with English terms, use Vietnamese + English terms. Auto can lock onto early English and decode later Vietnamese as English.
          </p>
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
          <label className="input-label">Visual Analysis Mode</label>
          <select className="form-select" value={visualMode} onChange={e => setVisualMode(e.target.value)}>
            <option value="Transcript only: skip slides/keyframes">Transcript only</option>
            <option value="Fast: capture keyframes, no OCR">Fast Keyframes (No OCR)</option>
            <option value="Full: capture keyframes + OCR">Full Keyframes + OCR</option>
            <option value="VLM: keyframes + image understanding">AI Keyframes + Image Understanding</option>
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
            checked={diarizationToggle}
            onChange={e => setDiarizationToggle(e.target.checked)}
          />
          Enable Speaker Diarization
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
            <button className={`tab-btn ${activeTab === 'evaluation' ? 'active' : ''}`} onClick={() => setActiveTab('evaluation')}>
              Evaluation Lab
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
                        <button className="btn-secondary" onClick={handleIngestPdf} title="Ingest exported PDF into chat context">
                          PDF {'->'} RAG
                        </button>
                        <button className="btn-secondary" onClick={() => downloadFromEndpoint('/export/quiz')} title="Download quiz bank JSON">
                          Quiz JSON
                        </button>
                        <button className="btn-secondary" onClick={() => downloadFromEndpoint('/export/anki')} title="Download Anki TSV">
                          Anki TSV
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
                            {block.segment_count || 0} segments / {block.word_count || 0} words
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
                      key={activeVideoSrc}
                      ref={videoRef} 
                      src={activeVideoSrc} 
                      controls 
                      preload="metadata"
                      className="html5-player"
                      onLoadedMetadata={() => setVideoLoadError(null)}
                      onError={() => setVideoLoadError('Video playback is not available. The backend may be down or the local video file is missing.')}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onSeeked={handleVideoTimeUpdate}
                    />
                    {videoLoadError && (
                      <div className="media-error-panel">
                        {videoLoadError}
                      </div>
                    )}
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
                          const speakerRole = formatSpeakerRole(seg);
                          const speakerLabel = formatSpeakerLabel(seg);
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
                              <div className="row-speaker">
                                {!(seg.speaker_source === 'role_cue_fallback' || String(seg.speaker || '').endsWith('_CUE')) && (
                                  <>
                                    <span className={`speaker-chip ${speakerRole}`}>{speakerLabel}</span>
                                    <span className="speaker-role">{speakerRole}</span>
                                  </>
                                )}
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
                              src={slide.image_path.startsWith('http') ? slide.image_path : `${API_BASE}/keyframes/${slide.image_path.split(/[/\\]/).pop()}`} 
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
                              <strong>Visual AI interpretation:</strong>
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
                          <ChatMessageContent content={msg.content} mode={msg.mode} />
                        </div>
                        {msg.role === 'assistant' && (
                          <div className="chat-citation-area">
                            {msg.engine && <span className="chat-engine">{msg.engine}{msg.latency_ms ? ` · ${msg.latency_ms}ms` : ''}</span>}
                            {(msg.citations || []).map((citation, cidx) => (
                              <button
                                key={`${citation.label}-${cidx}`}
                                className="citation-chip"
                                onClick={() => typeof citation.start === 'number' ? handleSeekVideo(citation.start) : undefined}
                                disabled={typeof citation.start !== 'number'}
                                title={citation.snippet || citation.label}
                              >
                                {citation.timestamp && citation.timestamp !== 'report' ? citation.timestamp : 'report'} · {citation.title || citation.source || 'source'}
                              </button>
                            ))}
                            {msg.message_id && (
                              <div className="feedback-row">
                                <button onClick={() => handleChatFeedback(msg.message_id, 'correct')}>Correct</button>
                                <button onClick={() => handleChatFeedback(msg.message_id, 'incomplete')}>Incomplete</button>
                                <button onClick={() => handleChatFeedback(msg.message_id, 'hallucinated')}>Hallucinated</button>
                              </div>
                            )}
                          </div>
                        )}
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
                  <div
                    className={`chat-mode-picker ${chatModeOpen ? 'open' : ''} ${chatLoading ? 'disabled' : ''}`}
                    ref={chatModeRef}
                  >
                    <button
                      type="button"
                      className="chat-mode-trigger"
                      onClick={() => !chatLoading && setChatModeOpen(open => !open)}
                      disabled={chatLoading}
                      aria-haspopup="listbox"
                      aria-expanded={chatModeOpen}
                    >
                      <span className="chat-mode-current">
                        <span>Query mode</span>
                        <strong>{selectedChatMode.label}</strong>
                      </span>
                      <ChevronDown size={16} />
                    </button>
                    {chatModeOpen && (
                      <div className="chat-mode-menu" role="listbox">
                        {CHAT_MODE_OPTIONS.map(option => (
                          <button
                            key={option.value}
                            type="button"
                            className={`chat-mode-option ${chatMode === option.value ? 'selected' : ''}`}
                            role="option"
                            aria-selected={chatMode === option.value}
                            onClick={() => {
                              setChatMode(option.value);
                              setChatModeOpen(false);
                            }}
                          >
                            <span className="chat-mode-option-dot" />
                            <span className="chat-mode-option-copy">
                              <strong>{option.label}</strong>
                              <small>{option.description}</small>
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
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

            {activeTab === 'evaluation' && (
              <div className="evaluation-grid">
                <div className="glass-panel eval-card">
                  <div className="eval-card-header">
                    <h3>Pipeline Metrics</h3>
                    <button className="btn-secondary" onClick={fetchEvaluationSummary}>Refresh</button>
                  </div>
                  <div className="metric-grid">
                    {Object.entries(evaluationSummary?.metrics || {}).map(([key, value]) => (
                      <div key={key} className="metric-tile">
                        <span>{key.replace(/_/g, ' ')}</span>
                        <strong>{formatMetricValue(value)}</strong>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-panel eval-card" style={{ gridColumn: 'span 2' }}>
                  <h3>Transcript Quality & Comparison</h3>
                  {evaluationSummary?.transcript_quality?.available ? (
                    <>
                      <div className="metric-grid" style={{ marginBottom: '1rem' }}>
                        <div className="metric-tile"><span>WER</span><strong>{evaluationSummary.transcript_quality.wer !== null ? `${Math.round(evaluationSummary.transcript_quality.wer * 100)}%` : 'n/a'}</strong></div>
                        <div className="metric-tile"><span>CER</span><strong>{evaluationSummary.transcript_quality.cer !== null ? `${Math.round(evaluationSummary.transcript_quality.cer * 100)}%` : 'n/a'}</strong></div>
                        <div className="metric-tile"><span>Quality grade</span><strong>{evaluationSummary.transcript_quality.grade}</strong></div>
                        <div className="metric-tile"><span>Reference words</span><strong>{evaluationSummary.transcript_quality.reference_words}</strong></div>
                        <div className="metric-tile"><span>Hypothesis words</span><strong>{evaluationSummary.transcript_quality.hypothesis_words}</strong></div>
                      </div>

                      <div style={{ marginTop: '1.25rem', overflowX: 'auto', maxHeight: '300px', overflowY: 'auto' }}>
                        <h4 style={{ marginBottom: '0.75rem', fontSize: '0.9rem', color: 'var(--text)' }}>Ground Truth vs ASR Comparison Samples</h4>
                        <table className="eval-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                          <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', opacity: 0.8 }}>
                              <th style={{ padding: '0.5rem' }}>Time</th>
                              <th style={{ padding: '0.5rem' }}>Reference (Ground Truth)</th>
                              <th style={{ padding: '0.5rem' }}>Hypothesis (ASR Output)</th>
                              <th style={{ padding: '0.5rem', textAlign: 'right' }}>WER</th>
                              <th style={{ padding: '0.5rem', textAlign: 'right' }}>CER</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(evaluationSummary.transcript_quality.samples || []).map((sample: any, idx: number) => (
                              <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', verticalAlign: 'top' }}>
                                <td style={{ padding: '0.5rem', whiteSpace: 'nowrap', color: 'var(--theme-color, #a855f7)' }}>
                                  {formatSecs(sample.start)}
                                </td>
                                <td style={{ padding: '0.5rem', color: '#10b981' }}>{sample.reference}</td>
                                <td style={{ padding: '0.5rem', color: '#f59e0b' }}>{sample.hypothesis}</td>
                                <td style={{ padding: '0.5rem', textAlign: 'right', fontWeight: 'bold' }}>
                                  {sample.wer !== null ? `${Math.round(sample.wer * 100)}%` : 'n/a'}
                                </td>
                                <td style={{ padding: '0.5rem', textAlign: 'right', fontWeight: 'bold' }}>
                                  {sample.cer !== null ? `${Math.round(sample.cer * 100)}%` : 'n/a'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                        <label className="btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', cursor: 'pointer', gap: '0.5rem' }}>
                          <Upload size={16} />
                          Upload New Reference Transcript (.srt, .vtt, .txt)
                          <input type="file" accept=".srt,.vtt,.txt" onChange={handleReferenceTranscriptUpload} style={{ display: 'none' }} />
                        </label>
                      </div>
                    </>
                  ) : (
                    <div>
                      <p className="eval-muted" style={{ marginBottom: '1rem' }}>
                        {evaluationSummary?.transcript_quality?.reason || 'Upload a Teams VTT/SRT transcript as reference to compute WER/CER.'}
                      </p>
                      <label className="btn-primary" style={{ display: 'inline-flex', alignItems: 'center', cursor: 'pointer', gap: '0.5rem' }}>
                        <Upload size={16} />
                        Upload Reference Transcript
                        <input type="file" accept=".srt,.vtt,.txt" onChange={handleReferenceTranscriptUpload} style={{ display: 'none' }} />
                      </label>
                    </div>
                  )}
                </div>

                <div className="glass-panel eval-card">
                  <h3>Visual Understanding Check</h3>
                  <p className="eval-muted">{evaluationSummary?.vlm_benchmark?.benchmark_engine || 'Run visual analysis to build a visual-grounding rubric.'}</p>
                  <div className="metric-grid">
                    <div className="metric-tile"><span>Slides evaluated</span><strong>{evaluationSummary?.vlm_benchmark?.slides_evaluated ?? 0}</strong></div>
                    <div className="metric-tile"><span>OCR coverage</span><strong>{evaluationSummary?.vlm_benchmark?.ocr_coverage ?? 0}</strong></div>
                    <div className="metric-tile"><span>Image analysis coverage</span><strong>{evaluationSummary?.vlm_benchmark?.vlm_coverage ?? 0}</strong></div>
                    <div className="metric-tile"><span>Fused visual confidence</span><strong>{evaluationSummary?.vlm_benchmark?.avg_ocr_vlm_confidence ?? 0}</strong></div>
                    <div className="metric-tile"><span>Strong evidence</span><strong>{evaluationSummary?.vlm_benchmark?.strong_visual_evidence ?? 0}</strong></div>
                    <div className="metric-tile"><span>Usable evidence</span><strong>{evaluationSummary?.vlm_benchmark?.usable_visual_evidence ?? 0}</strong></div>
                  </div>
                </div>

                <div className="glass-panel eval-card">
                  <h3>Ablation Snapshot</h3>
                  <p className="eval-muted">{evaluationSummary?.ablation?.engine || 'Run analysis to build ablation data.'}</p>
                  <div className="speaker-list">
                    {(evaluationSummary?.ablation?.rows || []).map((row: any) => (
                      <div className="speaker-row" key={row.mode}>
                        <span>{formatAblationMode(row.mode)}</span>
                        <strong>{row.confidence}</strong>
                        <em>{row.evidence_units} units</em>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-panel eval-card">
                  <h3>Speaker Role Pass & Mapping</h3>
                  <p className="eval-muted">{evaluationSummary?.speaker_roles?.engine || 'No speaker role data yet.'}</p>
                  {evaluationSummary?.speaker_roles?.diarization_status && (
                    <div className="speaker-row" style={{ marginBottom: '1rem' }}>
                      <span>{evaluationSummary.speaker_roles.diarization_status.provider || 'diarization'}</span>
                      <strong>{evaluationSummary.speaker_roles.diarization_status.status || 'unknown'}</strong>
                      <em>{evaluationSummary.speaker_roles.diarization_status.device || evaluationSummary.speaker_roles.diarization_status.reason || ''}</em>
                    </div>
                  )}
                  <div className="speaker-list">
                    {(evaluationSummary?.speaker_roles?.speakers || []).map((speaker: any, idx: number) => (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255, 255, 255, 0.04)', gap: '1rem' }}>
                        <div style={{ flex: '1', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                          <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{speaker.speaker}</span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{speaker.role} &bull; {speaker.segments} segments</span>
                        </div>
                        <input
                          type="text"
                          className="input-field"
                          placeholder="Rename..."
                          value={speakerMapping[speaker.speaker] || ''}
                          onChange={(e) => setSpeakerMapping({
                            ...speakerMapping,
                            [speaker.speaker]: e.target.value
                          })}
                          style={{
                            width: '120px',
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.8rem',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'white'
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  {evaluationSummary?.speaker_roles?.speakers && evaluationSummary.speaker_roles.speakers.length > 0 && (
                    <button 
                      className="btn-primary" 
                      onClick={handleSaveSpeakerMapping}
                      style={{ marginTop: '1rem', width: '100%' }}
                    >
                      Save Speaker Mappings
                    </button>
                  )}
                </div>

                <div className="glass-panel eval-card">
                  <h3>Exports & Readiness</h3>
                  <div className="eval-actions">
                    <button className="btn-secondary" onClick={handleIngestPdf}>Ingest PDF for Q&A</button>
                    <button className="btn-secondary" onClick={handleSyncStorage}>Sync Artifacts</button>
                    <button className="btn-secondary" onClick={() => downloadFromEndpoint('/export/quiz')}>Download Quiz JSON</button>
                    <button className="btn-secondary" onClick={() => downloadFromEndpoint('/export/anki')}>Download Anki TSV</button>
                    <button className="btn-secondary" onClick={handleBuildRegressionSet}>Build Regression Set</button>
                    <button className="btn-secondary" onClick={() => window.open(`${API_BASE}/evaluation/regression-set`, '_blank')}>Regression Set JSON</button>
                    <button className="btn-secondary" onClick={() => window.open(`${API_BASE}/deployment/readiness`, '_blank')}>Cloud/Worker Readiness JSON</button>
                  </div>
                  <p className="eval-muted">
                    Regression set: {evaluationSummary?.regression_set?.available
                      ? `${evaluationSummary.regression_set.cases} cases`
                      : evaluationSummary?.regression_set?.buildable_from_current_lecture
                        ? `not built yet (${evaluationSummary.regression_set.candidate_cases} candidate cases)`
                        : 'not available'}
                  </p>
                </div>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  );
}
