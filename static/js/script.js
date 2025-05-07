const socket = io();
let audio1Recorder, audio1Chunks = [];
let audio2Recorder, audio2Chunks = [];
let referenceRecorder, referenceChunks = [];
let lessons = [];

// Wavesurfer instances
const wavesurfer1 = WaveSurfer.create({
  container: '#waveform1',
  waveColor: '#FF6F61',
  progressColor: '#FF8A75',
  cursorColor: '#2E2E2E',
  height: 100
});
const wavesurfer2 = WaveSurfer.create({
  container: '#waveform2',
  waveColor: '#FF6F61',
  progressColor: '#FF8A75',
  cursorColor: '#2E2E2E',
  height: 100
});
const referenceWaveform = WaveSurfer.create({
  container: '#reference-waveform',
  waveColor: '#FF6F61',
  progressColor: '#FF8A75',
  cursorColor: '#2E2E2E',
  height: 100
});
const attemptWaveform = WaveSurfer.create({
  container: '#attempt-waveform',
  waveColor: '#FF6F61',
  progressColor: '#FF8A75',
  cursorColor: '#2E2E2E',
  height: 100
});

// Tab navigation
document.getElementById('analyze-tab').addEventListener('click', () =>
  toggleSection('analyze-section', 'lessons-section', 'analyze-tab', 'lessons-tab')
);
document.getElementById('lessons-tab').addEventListener('click', () => {
  toggleSection('lessons-section', 'analyze-section', 'lessons-tab', 'analyze-tab');
  renderLessons();
});

function toggleSection(show, hide, activeTab, inactiveTab) {
  document.getElementById(show).classList.remove('hidden');
  document.getElementById(hide).classList.add('hidden');
  document.getElementById(activeTab).classList.add('active');
  document.getElementById(inactiveTab).classList.remove('active');
}

// Analyze section elements
const uploadForm     = document.getElementById('uploadForm');
const audio1Upload   = document.getElementById('audio1-upload');
const audio1RecordBtn= document.getElementById('audio1-record-btn');
const audio2Upload   = document.getElementById('audio2-upload');
const audio2RecordBtn= document.getElementById('audio2-record-btn');
const analyzeBtn     = document.getElementById('analyze-btn');
const loading        = document.getElementById('loading');
const logs           = document.getElementById('logs');
const result         = document.getElementById('result');
const similarityText = document.getElementById('similarity');
const transcription1 = document.getElementById('transcription1');
const transcription2 = document.getElementById('transcription2');
const metricsDiv     = document.getElementById('metrics');
const metricsList    = document.getElementById('metrics-list');
const comparisonImage= document.getElementById('comparisonImage');
const exportBtn      = document.getElementById('export-btn');

let audio1Ready = false, audio2Ready = false;
function updateAnalyzeButton() {
  analyzeBtn.disabled = !(audio1Ready && audio2Ready);
}

// Upload/record Audio1
audio1Upload.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) {
    wavesurfer1.loadBlob(file);
    audio1Chunks = [file];
    audio1Ready = true;
    updateAnalyzeButton();
  }
});
audio1RecordBtn.addEventListener('click', async () => {
  if (!audio1Recorder || audio1Recorder.state === 'inactive') {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audio1Recorder = new MediaRecorder(stream);
    audio1Chunks = [];
    audio1Recorder.ondataavailable = ev => audio1Chunks.push(ev.data);
    audio1Recorder.onstop = () => {
      const blob = new Blob(audio1Chunks, { type: 'audio/wav' });
      wavesurfer1.loadBlob(blob);
      audio1RecordBtn.textContent = 'Record';
      audio1Ready = true;
      updateAnalyzeButton();
    };
    audio1Recorder.start();
    audio1RecordBtn.textContent = 'Stop Recording';
  } else {
    audio1Recorder.stop();
  }
});

// Upload/record Audio2
audio2Upload.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) {
    wavesurfer2.loadBlob(file);
    audio2Chunks = [file];
    audio2Ready = true;
    updateAnalyzeButton();
  }
});
audio2RecordBtn.addEventListener('click', async () => {
  if (!audio2Recorder || audio2Recorder.state === 'inactive') {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audio2Recorder = new MediaRecorder(stream);
    audio2Chunks = [];
    audio2Recorder.ondataavailable = ev => audio2Chunks.push(ev.data);
    audio2Recorder.onstop = () => {
      const blob = new Blob(audio2Chunks, { type: 'audio/wav' });
      wavesurfer2.loadBlob(blob);
      audio2RecordBtn.textContent = 'Record';
      audio2Ready = true;
      updateAnalyzeButton();
    };
    audio2Recorder.start();
    audio2RecordBtn.textContent = 'Stop Recording';
  } else {
    audio2Recorder.stop();
  }
});

// Submit analyze form
uploadForm.addEventListener('submit', async e => {
  e.preventDefault();
  const formData = new FormData(uploadForm);
  if (audio1Chunks.length) formData.set('audio1', new Blob(audio1Chunks, { type: 'audio/wav' }), 'audio1.wav');
  if (audio2Chunks.length) formData.set('audio2', new Blob(audio2Chunks, { type: 'audio/wav' }), 'audio2.wav');

  loading.classList.remove('hidden');
  result.classList.add('hidden');
  logs.innerHTML = '';

  try {
    const response = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed');

    similarityText.textContent = `Similarity: ${data.similarity}%`;
    transcription1.innerHTML = `Audio 1: ${data.transcription_html}`;
    transcription2.textContent = `Audio 2: ${data.transcription2}`;

    if (data.image) {
      comparisonImage.src = `/get_image?ts=${Date.now()}`;
      comparisonImage.classList.remove('hidden');
    }
    if (data.metrics) {
      metricsList.innerHTML = Object.entries(data.metrics)
        .map(([k,v]) => `<li>${k}: ${v.toFixed(2)}%</li>`).join('');
      metricsDiv.classList.remove('hidden');
    }
    result.classList.remove('hidden');
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    loading.classList.add('hidden');
  }
});

// Export results JSON
exportBtn.addEventListener('click', () => {
  const resultData = {
    similarity: similarityText.textContent,
    transcriptions: [
      transcription1.textContent,
      transcription2.textContent
    ],
    metrics: Array.from(metricsList.children).map(li => li.textContent)
  };
  const blob = new Blob([JSON.stringify(resultData, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'speech_similarity_results.json';
  a.click();
  URL.revokeObjectURL(a.href);
});

// Lessons section
const lessonList     = document.getElementById('lesson-list');
const addLessonBtn   = document.getElementById('add-lesson-btn');
const lessonSearch   = document.getElementById('lesson-search');
const lessonView     = document.getElementById('lesson-view');
const lessonTitle    = document.getElementById('lesson-title');
const lessonCreate   = document.getElementById('lesson-create');
const lessonName     = document.getElementById('lesson-name');
const referenceAudio = document.getElementById('reference-audio');
const recordReferenceBtn = document.getElementById('record-reference-btn');
const saveLessonBtn      = document.getElementById('save-lesson-btn');
const transcriptionField  = document.getElementById('transcription');
const lessonProgress      = document.getElementById('lesson-progress');
const recordAttemptBtn    = document.getElementById('record-attempt-btn');
const compareBtn          = document.getElementById('compare-btn');
const progressLoading     = document.getElementById('progress-loading');
const progressLogs        = document.getElementById('progress-logs');
const latestScore         = document.getElementById('latest-score');
const scoreList           = document.getElementById('score-list');

let currentLessonId = null;

function renderLessons(filter='') {
  const filtered = lessons.filter(l =>
    l.name.toLowerCase().includes(filter.toLowerCase())
  );
  lessonList.innerHTML = filtered.length
    ? filtered.map(l => `
      <div class="lesson-item" data-id="${l.id}">
        <span class="lesson-name">${l.name}</span>
        <div class="lesson-progress">
          <div class="progress-bar" style="width: ${l.attempts.length? l.attempts[l.attempts.length-1].similarity:0}%"></div>
        </div>
        <button onclick="showLesson(${l.id})">View</button>
        <button onclick="deleteLesson(${l.id})" class="delete-btn">✕</button>
      </div>
    `).join('')
    : '<p class="empty-state">No lessons found.</p>';
}

addLessonBtn.addEventListener('click', () => {
  currentLessonId = null;
  lessonTitle.textContent = 'Create New Lesson';
  lessonList.classList.add('hidden');
  lessonView.classList.remove('hidden');
  lessonCreate.classList.remove('hidden');
  lessonProgress.classList.add('hidden');
  lessonName.value = '';
  referenceWaveform.empty();
  transcriptionField.value = '';
  saveLessonBtn.disabled = true;
  referenceChunks = [];
  addLessonBtn.classList.add('hidden');
  lessonSearch.classList.add('hidden');
});

lessonSearch.addEventListener('input', e => renderLessons(e.target.value));

window.showLesson = id => {
  currentLessonId = id;
  const l = lessons.find(x => x.id===id);
  lessonTitle.textContent = l.name;
  lessonList.classList.add('hidden');
  lessonView.classList.remove('hidden');
  lessonCreate.classList.add('hidden');
  lessonProgress.classList.remove('hidden');
  latestScore.textContent = l.attempts.length? `Latest: ${l.attempts[l.attempts.length-1].similarity}%` : '';
  scoreList.innerHTML = l.attempts.map(a=>`<li>${a.similarity}%</li>`).join('');
};

window.backToList = () => {
  lessonView.classList.add('hidden');
  lessonList.classList.remove('hidden');
  renderLessons(lessonSearch.value);
  addLessonBtn.classList.remove('hidden');
  lessonSearch.classList.remove('hidden');
};

window.deleteLesson = async id => {
  const res = await fetch(`/delete_lesson/${id}`, { method: 'POST' });
  const data = await res.json();
  if (data.success) {
    lessons = lessons.filter(l=>l.id!==id);
    renderLessons(lessonSearch.value);
  }
};

// Reference audio upload/record
referenceAudio.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) {
    referenceWaveform.loadBlob(file);
    referenceChunks = [file];
    updateSaveBtn();
  }
});
recordReferenceBtn.addEventListener('click', async ()=>{
  if (!referenceRecorder||referenceRecorder.state==='inactive') {
    const s=await navigator.mediaDevices.getUserMedia({audio:true});
    referenceRecorder=new MediaRecorder(s);
    referenceChunks=[];
    referenceRecorder.ondataavailable=ev=>referenceChunks.push(ev.data);
    referenceRecorder.onstop=()=>{
      const b=new Blob(referenceChunks,{type:'audio/wav'});
      referenceWaveform.loadBlob(b);
      recordReferenceBtn.textContent='Record Reference';
      updateSaveBtn();
    };
    referenceRecorder.start();
    recordReferenceBtn.textContent='Stop Recording';
  } else referenceRecorder.stop();
});

function updateSaveBtn() {
  saveLessonBtn.disabled=!(
    lessonName.value.trim() &&
    transcriptionField.value.trim() &&
    referenceChunks.length
  );
}

lessonName.addEventListener('input', updateSaveBtn);
transcriptionField.addEventListener('input', updateSaveBtn);

saveLessonBtn.addEventListener('click', async ()=>{
  const fd=new FormData();
  fd.append('name', lessonName.value);
  fd.append('transcription', transcriptionField.value);
  fd.append('reference_audio', new Blob(referenceChunks,{type:'audio/wav'}),'reference.wav');

  try {
    const res=await fetch('/add_lesson',{method:'POST',body:fd});
    const data=await res.json();
    if (!res.ok||!data.success) throw new Error(data.error||'Failed to save lesson');
    currentLessonId = data.id;
    lessons.push({
      id: data.id,
      name: lessonName.value,
      transcription: transcriptionField.value,
      reference_audio: URL.createObjectURL(new Blob(referenceChunks)),
      attempts: []
    });
    lessonTitle.textContent=lessonName.value;
    lessonCreate.classList.add('hidden');
    lessonProgress.classList.remove('hidden');
    latestScore.textContent='';
    scoreList.innerHTML='';
    renderLessons(lessonSearch.value);
    addLessonBtn.classList.remove('hidden');
    lessonSearch.classList.remove('hidden');
  } catch(err){
    alert(`Error saving lesson: ${err.message}`);
  }
});

// Record & compare attempts
recordAttemptBtn.addEventListener('click', async ()=>{
  if (!audio2Recorder||audio2Recorder.state==='inactive') {
    const s=await navigator.mediaDevices.getUserMedia({audio:true});
    audio2Recorder=new MediaRecorder(s);
    audio2Chunks=[];
    audio2Recorder.ondataavailable=ev=>audio2Chunks.push(ev.data);
    audio2Recorder.onstop=()=>{
      attemptWaveform.loadBlob(new Blob(audio2Chunks,{type:'audio/wav'}));
      recordAttemptBtn.textContent='Record Attempt';
    };
    audio2Recorder.start();
    recordAttemptBtn.textContent='Stop Recording';
  } else audio2Recorder.stop();
});

compareBtn.addEventListener('click',async()=>{
  const fd=new FormData();
  fd.append('attempt', new Blob(audio2Chunks,{type:'audio/wav'}),'attempt.wav');
  progressLoading.classList.remove('hidden');
  progressLogs.innerHTML='';
  try {
    const res=await fetch(`/update_lesson/${currentLessonId}`,{method:'POST',body:fd});
    const data=await res.json();
    if (!res.ok) throw new Error(data.error||'Analysis failed');
    const l=lessons.find(x=>x.id===currentLessonId);
    l.attempts.push({similarity:parseFloat(data.similarity)});
    latestScore.textContent=`Latest: ${data.similarity}%`;
    scoreList.innerHTML = l.attempts.map(a=>`<li>${a.similarity}%</li>`).join('');
    document.querySelector(`.lesson-item[data-id="${currentLessonId}"] .progress-bar`)
      .style.width=`${data.similarity}%`;
  } catch(err){
    alert(`Error: ${err.message}`);
  } finally {
    progressLoading.classList.add('hidden');
  }
});

// Socket logs
socket.on('log', data=>{
  const p=document.createElement('p');
  p.textContent=data.message;
  (lessonView.classList.contains('hidden')?logs:progressLogs).appendChild(p);
});

renderLessons();
