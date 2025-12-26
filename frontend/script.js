// Simple i18n dictionary
const dict = {
  en: {
    subtitle: " Code Vulnerability Detection",
    heroTitle: "Secure Your Code. Detect Threats",
    heroDesc: "Paste your code or upload a file to analyze for potential security vulnerabilities using our deep learning model.",
    editorTitle: "Code Input",
    modelName: "Deep Learning Model",
    uploadBtn: "Upload File",
    clearBtn: "Clear",
    analyzeBtn: "Analyze Code",
    resultTitle: "Security Analysis Result",
    verdictSafe: "Code is Safe",
    verdictMal: "Malicious Code Detected",
    verdictUnknown: "Awaiting Analysis",
    riskLabel: "Risk Level",
    safeLabel: "Safe",
    vulnerableLabel: "Vulnerable",
    confidenceLabel: "Confidence:",
    analysisError: "Analysis Error",
    serverError: "Unable to reach server",
    checkEndpoint: "Check the API endpoint and server status.",
    securityIssue: "Security Issue",
    enterCode: "Please enter code to analyze",
    analyzing: "Analyzing...",
    timeLabel: "Time:"  // Added localized label for time metadata
  }
};
let lang = 'en'; // Force English

// Language toggling
const applyLang = () => {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.textContent = dict[lang][key] || el.textContent;
  });
  document.documentElement.setAttribute('dir', 'ltr');
  document.getElementById('code').setAttribute('placeholder', "// Paste your code here or drop a file (e.g., C, C++)");
};

applyLang();

// Editor: line numbers and drag-drop
const codeEl = document.getElementById('code');
const gutterEl = document.getElementById('gutter');
const editorEl = document.getElementById('editor');

const updateLines = () => {
  const lines = codeEl.value.split('\n').length;
  let out = '';
  for(let i=1;i<=lines;i++) out += i + (i<lines?'\n':'');
  gutterEl.textContent = out || '1';
};
codeEl.addEventListener('input', updateLines);
codeEl.addEventListener('paste', () => setTimeout(updateLines, 0));
updateLines();

// Drag & drop
let droppedFile = null;  // variable carry draged file
['dragenter','dragover'].forEach(evt =>
  editorEl.addEventListener(evt, e => { e.preventDefault(); editorEl.style.outline='1px dashed #2a3b52'; })
);
['dragleave','drop'].forEach(evt =>
  editorEl.addEventListener(evt, e => { e.preventDefault(); editorEl.style.outline='none'; })
);
editorEl.addEventListener('drop', async e => {
  const file = e.dataTransfer.files?.[0];
  if(file){
    droppedFile = file; 
    const text = await file.text();
    codeEl.value = text;
    updateLines();
  }
});


// File input
document.getElementById('file').addEventListener('change', async e => {
  const file = e.target.files?.[0];
  if(file){
    const text = await file.text();
    codeEl.value = text;
    updateLines();
  }
});

// Clear
document.getElementById('clearBtn').onclick = () => {
  codeEl.value = '';
  updateLines();

  // Reset dropped file and file input
  droppedFile = null;
  document.getElementById('file').value = '';
  
  renderResult(null);
};

// Analyze: call your backend
const analyzeBtn = document.getElementById('analyzeBtn');
analyzeBtn.onclick = async () => {
  setLoading(true);
  try {
    const formData = new FormData();
    const fileInput = document.getElementById('file'); 
    const code = codeEl.value.trim(); 

    if (droppedFile) {
      formData.append("file", droppedFile);
    } else if (fileInput.files[0]) {
      formData.append("file", fileInput.files[0]);
    } else if (code) {
      formData.append("code", code);
    }

    // If nothing provided
    if (!code && !fileInput.files[0] && !droppedFile) {
      flashChip("Please enter code");
      setLoading(false);
      return;
    }
    // Send request to FastAPI backend
    const res = await fetch("http://" + window.location.hostname + ":8000/predict", {
      method:'POST',
      body: formData
    });

    const data = await res.json();
    renderResult(data);
  }catch(err){
    renderError(err);
  }finally{
    setLoading(false);
  }
};

// UI helpers
const verdictChip = document.getElementById('verdictChip');
const riskBar = document.getElementById('riskBar');
const issuesEl = document.getElementById('issues');
const modelMeta = document.getElementById('modelMeta');
const timeMeta = document.getElementById('timeMeta');

// New elements for status card
const safeStatus = document.getElementById('safeStatus');
const vulnerableStatus = document.getElementById('vulnerableStatus');
const safeConfidence = document.getElementById('safeConfidence');
const vulnerableConfidence = document.getElementById('vulnerableConfidence');

function renderResult(data){
  console.log(data)

  issuesEl.innerHTML = '';
  modelMeta.textContent = '—';
  timeMeta.textContent = '';

  // Reset status card
  safeStatus.classList.remove('active');
  vulnerableStatus.classList.remove('active');
  safeConfidence.textContent = '';
  vulnerableConfidence.textContent = '';

  if(!data){
    verdictChip.className='chip';
    verdictChip.textContent = dict[lang].verdictUnknown;
    riskBar.style.width = '0%';
    return;
  }
  const verdict = String(data.verdict || '').toLowerCase();
  const risk = Math.max(0, Math.min(1, Number(data.risk ?? 0)));
  
  // Calculate confidence (assuming risk is the probability of being malicious)
  const vulnerabilityConfidence = (risk * 100).toFixed(0);
  const safeConfidenceValue = ((1 - risk) * 100).toFixed(0);

  // Update status card
  safeConfidence.textContent = `${dict[lang].confidenceLabel} ${safeConfidenceValue}%`;
  vulnerableConfidence.textContent = `${dict[lang].confidenceLabel} ${vulnerabilityConfidence}%`;

  if (verdict === 'malicious') {
    vulnerableStatus.classList.add('active');
  } else if (verdict === 'safe') {
    safeStatus.classList.add('active');
  } else {
    // If unknown, no status is active
  }

  // Existing verdict chip logic
  verdictChip.className = 'chip ' + (verdict === 'malicious' ? 'malicious' : verdict === 'safe' ? 'safe' : '');
  verdictChip.textContent = verdict === 'malicious' ? dict[lang].verdictMal :
                            verdict === 'safe' ? dict[lang].verdictSafe :
                            dict[lang].verdictUnknown;

  riskBar.style.width = (risk*100).toFixed(0) + '%';

  (data.issues || []).forEach(issue => {
    const div = document.createElement('div');
    div.className='issue';
    const title = document.createElement('h4');
    title.textContent = issue.title || 'Security issue';
    const detail = document.createElement('p');
    detail.textContent = issue.detail || '';
    div.appendChild(title); div.appendChild(detail);
    issuesEl.appendChild(div);
  });
  modelMeta.textContent = (data.model || '—');
  timeMeta.textContent = data.timeMs ? (dict[lang].timeLabel + ' ' + data.timeMs + ' ms') : '';}


function renderError(err){
  verdictChip.className='chip malicious';
  verdictChip.textContent = dict[lang].analysisError;
  riskBar.style.width = '0%';
  issuesEl.innerHTML = '';
  const div = document.createElement('div');
  div.className='issue';
  const h4 = document.createElement('h4');
  h4.textContent = dict[lang].serverError;
  const p = document.createElement('p');
  p.textContent = err?.message || dict[lang].checkEndpoint;
  div.appendChild(h4); div.appendChild(p);
  issuesEl.appendChild(div);
  modelMeta.textContent = '—';
  timeMeta.textContent = '';

  // Reset status card on error
  safeStatus.classList.remove('active');
  vulnerableStatus.classList.remove('active');
  safeConfidence.textContent = '';
  vulnerableConfidence.textContent = '';
}

function setLoading(loading){
  analyzeBtn.disabled = loading;
  analyzeBtn.textContent = loading ? dict[lang].analyzing : dict[lang].analyzeBtn;
}

function flashChip(text){
  verdictChip.className='chip';
  verdictChip.textContent = text;
  setTimeout(()=>{ verdictChip.textContent = dict[lang].verdictUnknown; }, 2000);
}
