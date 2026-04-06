const audioInput = document.getElementById("audioInput");
const fileLabel = document.getElementById("fileLabel");
const titleInput = document.getElementById("titleInput");
const artistInput = document.getElementById("artistInput");
const uploadButton = document.getElementById("uploadButton");
const searchButton = document.getElementById("searchButton");
const recordButton = document.getElementById("recordButton");
const recordStatus = document.getElementById("recordStatus");
const endpointLabel = document.getElementById("endpointLabel");
const output = document.getElementById("output");

let selectedFile = null;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
const allowedExtensions = ["mp3", "wav"];

function getFileExtension(fileName) {
  const parts = (fileName || "").toLowerCase().split(".");
  return parts.length > 1 ? parts[parts.length - 1] : "";
}

function isSupportedAudioFile(file) {
  return allowedExtensions.includes(getFileExtension(file.name));
}

function interleaveChannels(channelData) {
  if (channelData.length === 1) {
    return channelData[0];
  }

  const left = channelData[0];
  const right = channelData[1];
  const interleaved = new Float32Array(left.length + right.length);
  let offset = 0;

  for (let i = 0; i < left.length; i += 1) {
    interleaved[offset] = left[i];
    interleaved[offset + 1] = right[i];
    offset += 2;
  }

  return interleaved;
}

function encodeWav(audioBuffer) {
  const numberOfChannels = Math.min(audioBuffer.numberOfChannels, 2);
  const sampleRate = audioBuffer.sampleRate;
  const channelData = [];

  for (let channel = 0; channel < numberOfChannels; channel += 1) {
    channelData.push(audioBuffer.getChannelData(channel));
  }

  const samples = interleaveChannels(channelData);
  const byteRate = sampleRate * numberOfChannels * 2;
  const blockAlign = numberOfChannels * 2;
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

async function convertBlobToWav(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("This browser does not support audio conversion to WAV.");
  }

  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContextClass();

  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    return encodeWav(audioBuffer);
  } finally {
    await audioContext.close();
  }
}

function setOutput(mainText, subText) {
  output.innerHTML = "";

  const main = document.createElement("p");
  main.className = "result-main";
  main.textContent = mainText;

  output.appendChild(main);

  if (subText) {
    const subLines = subText.split("\n");
    for (const line of subLines) {
      const sub = document.createElement("p");
      sub.className = "result-sub";
      sub.textContent = line;
      output.appendChild(sub);
    }
  }
}

function setBusy(isBusy) {
  uploadButton.disabled = isBusy;
  searchButton.disabled = isBusy;
  recordButton.disabled = isBusy && recordButton.textContent !== "Stop recording";
}

function setSelectedFile(file) {
  selectedFile = file;
  fileLabel.textContent = file ? `${file.name} (${Math.round(file.size / 1024)} KB)` : "Drop or choose an audio file";
}

function buildFormData(includeMetadata) {
  const file = selectedFile || audioInput.files[0];
  if (!file) {
    throw new Error("Choose or record an audio file first.");
  }
  if (!isSupportedAudioFile(file)) {
    throw new Error("Only MP3 and WAV files are supported.");
  }

  const formData = new FormData();
  formData.append("audio", file);
  if (includeMetadata) {
    if (titleInput.value.trim()) {
      formData.append("title", titleInput.value.trim());
    }
    if (artistInput.value.trim()) {
      formData.append("artist", artistInput.value.trim());
    }
  }

  return formData;
}

async function submitAction(action) {
  if (action === "upload") {
    const title = titleInput.value.trim();
    const artist = artistInput.value.trim();

    if (!title) {
      setOutput("Missing song title", "Please enter a song title before uploading.");
      return;
    }
    if (!artist) {
      setOutput("Missing artist name", "Please enter an artist name before uploading.");
      return;
    }
  }

  const endpoint = action === "upload" ? "/songs/upload" : "/songs/search";
  setBusy(true);

  try {
    const formData = buildFormData(action === "upload");
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload?.detail ? JSON.stringify(payload.detail, null, 2) : response.statusText;
      throw new Error(message || "Request failed");
    }

    if (action === "upload") {
      const songTitle = payload.title;
      const songArtist = payload.artist;
      const duration = typeof payload.duration === "number" ? `${payload.duration.toFixed(2)} s` : "N/A";
      const artistLine = songArtist ? `Artist: ${songArtist}` : "";
      const subtitle = artistLine ? `${artistLine} | Duration: ${duration}` : `Duration: ${duration}`;
      setOutput(`Uploaded: ${songTitle}`, subtitle);
    } else {
      const matched = payload.title;
      const timeTaken = typeof payload.time_taken_s === "number" ? `Identified in ${payload.time_taken_s.toFixed(3)} s` : "N/A";

      if (matched) {
        const songTitle = payload.title;
        const songArtist = payload.artist;
        const artistLine = songArtist ? `Artist: ${songArtist}` : "";
        const subtitle = artistLine ? `${artistLine}\n${timeTaken}` : timeTaken;
        setOutput(`${songTitle}`, subtitle);
      } else {
        setOutput("No confident match found.", timeTaken);
      }
    }
  } catch (error) {
    setOutput("Request failed", error.message || String(error));
  } finally {
    setBusy(false);
  }
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordButton.textContent = "Start recording";
    recordStatus.textContent = "Finalizing recording...";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });

      try {
        const wavBlob = await convertBlobToWav(blob);
        const file = new File([wavBlob], `recording-${Date.now()}.wav`, { type: "audio/wav" });
        setSelectedFile(file);
        audioInput.value = "";
        recordStatus.textContent = "Recording saved as WAV";
      } catch (error) {
        recordStatus.textContent = "Recording conversion failed";
        setOutput("Could not convert recording", error.message || "Please upload an MP3 or WAV file.");
      } finally {
        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
        }
      }
    };

    mediaRecorder.start();
    recordButton.textContent = "Stop recording";
    recordStatus.textContent = "Recording...";
  } catch (error) {
    recordStatus.textContent = "Microphone unavailable";
    setOutput("Microphone unavailable", error.message || "Unable to access the microphone.");
  }
}

audioInput.addEventListener("change", () => {
  const file = audioInput.files[0] || null;
  if (file && !isSupportedAudioFile(file)) {
    audioInput.value = "";
    setSelectedFile(null);
    setOutput("Unsupported audio format", "Please select an MP3 or WAV file.");
    return;
  }
  setSelectedFile(file);
});

uploadButton.addEventListener("click", () => submitAction("upload"));
searchButton.addEventListener("click", () => submitAction("search"));
recordButton.addEventListener("click", toggleRecording);

setOutput("Select an audio file or record a clip.", "Then choose Upload song or Search song.");