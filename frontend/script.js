/**
 * Viana RAG System - Frontend JavaScript
 * Main client-side logic for chat interface
 * WITH INPUT DISABLE DURING STREAMING
 */

const API_BASE_URL = "https://nisi-lakia-hyperprognathous.ngrok-free.dev"; // Use relative URLs - goes through Nginx proxy
const STORAGE_KEY = "novira_session";

class NoviraChat {
  constructor() {
    this.session_id = this.getSessionId();
    this.files = [];
    this.isLoading = false;
    this.isStreaming = false; // NEW: Track streaming state
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.showInitialLoading();
    this.loadUploadedFiles().then(() => {
      this.hideInitialLoading();
    });
    console.log("✅ Viana is ready");
  }

  showInitialLoading() {
    const filesList = document.getElementById("filesList");
    filesList.innerHTML = `
      <li class="loading-item">
        <div class="loading-spinner"></div>
        <span>Loading files...</span>
      </li>
    `;
  }

  hideInitialLoading() {
    // Loading will be replaced by loadUploadedFiles
  }

  setupEventListeners() {
    // File input change event
    document
      .getElementById("fileInput")
      .addEventListener("change", (e) => this.handleFileUpload(e));

    // Chat form submit event
    document
      .getElementById("chatForm")
      .addEventListener("submit", (e) => this.handleSendMessage(e));

    // Clear button click event
    document
      .getElementById("clearBtn")
      .addEventListener("click", () => this.clearFiles());

    // Auto-update status when input is focused
    document.getElementById("messageInput").addEventListener("focus", () => {
      this.updateStatus("Active");
    });

    // Handle Enter and Shift+Enter for multiline input
    document.getElementById("messageInput").addEventListener("keydown", (e) => {
      // Shift+Enter: Allow new line (default behavior)
      if (e.key === "Enter" && e.shiftKey) {
        // Let default behavior happen (adds new line)
        // Textarea will auto-expand
        return;
      }

      // Enter without Shift: Submit form
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();

        // Prevent sending while streaming
        if (this.isStreaming) {
          console.log("Please wait for the current response to complete");
          return;
        }

        // Create a synthetic event and call handleSendMessage
        const syntheticEvent = new Event("submit", {
          bubbles: true,
          cancelable: true,
        });
        syntheticEvent.preventDefault = () => {};
        this.handleSendMessage(syntheticEvent);
      }
    });

    // Auto-resize textarea as user types
    const messageInput = document.getElementById("messageInput");
    messageInput.addEventListener("input", () => {
      this.autoResizeTextarea(messageInput);
    });
  }

  getSessionId() {
    // Retrieve or create a unique session ID
    let sessionId = localStorage.getItem(STORAGE_KEY);
    if (!sessionId) {
      sessionId =
        "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
      localStorage.setItem(STORAGE_KEY, sessionId);
    }
    return sessionId;
  }

  updateStatus(status) {
    // Update the status indicator text
    document.getElementById("statusText").textContent = status;
  }

  // Auto-resize textarea based on content
  autoResizeTextarea(textarea) {
    // Reset height to recalculate
    textarea.style.height = "auto";

    // Calculate new height (max 200px)
    const newHeight = Math.min(textarea.scrollHeight, 200);

    // Set new height
    textarea.style.height = newHeight + "px";
  }

  // NEW: Disable input and button
  disableInput() {
    const input = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const footer = document.querySelector(".chat-footer");

    input.disabled = true;
    input.placeholder = "⏳ Waiting for response...";
    sendBtn.disabled = true;

    // Add visual feedback
    footer.classList.add("processing");

    this.isStreaming = true;
    this.updateStatus("Processing...");
  }

  // Enable input and button
  enableInput() {
    const input = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const footer = document.querySelector(".chat-footer");

    input.disabled = false;
    input.placeholder = "Type your question here... (Shift+Enter for new line)";
    sendBtn.disabled = false;

    // Remove visual feedback
    footer.classList.remove("processing");

    this.isStreaming = false;
    this.updateStatus("Active");

    // Reset textarea height
    input.style.height = "auto";

    // Focus back to input for better UX
    setTimeout(() => input.focus(), 100);
  }

  async handleFileUpload(event) {
    // Handle file selection and upload
    const files = event.target.files;
    if (files.length === 0) return;

    this.updateStatus("Uploading...");
    const statusDiv = document.getElementById("uploadStatus");
    statusDiv.textContent = "";

    // Upload each file
    for (let file of files) {
      try {
        this.updateStatus(`Uploading ${file.name}...`);

        // Create status element for this file
        const fileStatusDiv = document.createElement("div");
        fileStatusDiv.style.color = "orange";
        fileStatusDiv.textContent = `⏳ Uploading ${file.name}...`;
        statusDiv.appendChild(fileStatusDiv);

        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          this.files.push(file.name);

          // Update status to processing
          this.updateStatus(`Processing ${file.name}...`);

          // Show success message in green with chunk info if available
          fileStatusDiv.style.color = "#28a745";
          const chunkInfo = data.chunks_in_file ? ` (${data.chunks_in_file} chunks)` : '';
          fileStatusDiv.textContent = `✓ ${file.name}${chunkInfo}`;

          // Fade out and remove after 2 seconds
          setTimeout(() => {
            fileStatusDiv.style.transition = "opacity 0.5s ease-out";
            fileStatusDiv.style.opacity = "0";

            // Remove element after fade completes
            setTimeout(() => {
              fileStatusDiv.remove();
            }, 500);
          }, 2000);

          console.log(`✅ File uploaded: ${file.name}`);
        } else {
          // Show error in red with message from backend
          const errorData = await response.json().catch(() => ({}));
          const errorMsg = errorData.detail || errorData.message || "Failed";
          fileStatusDiv.style.color = "#dc3545";
          fileStatusDiv.textContent = `✗ ${file.name} - ${errorMsg}`;
        }
      } catch (error) {
        console.error(`Upload error: ${error}`);

        // Find and update the status div
        const fileStatusDivs = statusDiv.querySelectorAll("div");
        const lastDiv = fileStatusDivs[fileStatusDivs.length - 1];
        if (lastDiv) {
          lastDiv.style.color = "#dc3545";
          lastDiv.textContent = `✗ ${file.name} - Error: ${error.message}`;
        }
      }
    }

    // Refresh file list and reset status
    this.updateStatus("Loading files...");
    await this.loadUploadedFiles();

    // Add small delay to show "Ready" status
    setTimeout(() => {
      this.updateStatus("Ready");
    }, 500);

    // Remove welcome message if it exists
    const welcomeMsg = document.querySelector(".welcome-message");
    if (welcomeMsg) {
      welcomeMsg.remove();
    }
  }

  async loadUploadedFiles() {
    // Load list of uploaded files from backend
    try {
      const response = await fetch(`${API_BASE_URL}/uploaded-files`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });

      const data = await response.json();
      

      const filesListEl = document.getElementById("filesList");
      filesListEl.innerHTML = "";

      const files = Array.isArray(data.filesList) ? data.filesList : [];

      if (files.length === 0) {
        filesListEl.innerHTML = '<li class="empty">No files uploaded</li>';
      } else {
        files.forEach((file) => {
          const li = document.createElement("li");
          li.className = "file-item";

          const fileNameSpan = document.createElement("span");
          fileNameSpan.className = "file-name";
          fileNameSpan.textContent = `📄 ${file}`;

          const deleteBtn = document.createElement("button");
          deleteBtn.className = "file-delete-btn";
          deleteBtn.innerHTML = "×";
          deleteBtn.title = "Delete this file";
          deleteBtn.onclick = (e) => {
            e.stopPropagation();
            this.deleteFile(file);
          };

          li.appendChild(fileNameSpan);
          li.appendChild(deleteBtn);
          filesListEl.appendChild(li);
        });
      }

      this.files = files;
    } catch (error) {
      console.error("Error loading files:", error);
    }
  }

  async handleSendMessage(event) {
    // Handle user message submission with streaming
    event.preventDefault();

    const input = document.getElementById("messageInput");
    const message = input.value.trim();

    if (!message) return;

    // IMPORTANT: Check if already streaming
    if (this.isStreaming) {
      console.log("⏳ Already processing a question. Please wait...");
      return;
    }

    // Check if files are uploaded
    if (this.files.length === 0) {
      this.addMessage("Viana", "Please upload a file first 📄", true);
      return;
    }

    // Add user message to chat
    this.addMessage("You", message, false);
    input.value = "";

    // Disable input immediately
    this.disableInput();

    // Show loading indicator
    this.showLoadingIndicator();

    try {
      // Use streaming endpoint
      await this.handleStreamingResponse(message);
    } catch (error) {
      console.error("Error:", error);
      this.removeLoadingIndicator();
      this.addMessage("Viana", `Error: ${error.message}`, true);
    } finally {
      // IMPORTANT: Always re-enable input when done
      this.enableInput();
    }
  }

  async handleStreamingResponse(question) {
    // Handle Server-Sent Events (SSE) streaming response
    // Keep loading indicator until first token arrives

    let messageDiv = null;
    let contentDiv = null;
    let fullAnswer = "";
    let sources = [];
    let streamComplete = false;
    let firstTokenReceived = false;

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          session_id: this.session_id,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.message || "Streaming request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          streamComplete = true;
          break;
        }

        // Decode the chunk
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "token") {
                // On first token, remove loading and create message container
                if (!firstTokenReceived) {
                  firstTokenReceived = true;
                  this.removeLoadingIndicator();
                  messageDiv = this.createBotMessageContainer();
                  contentDiv = messageDiv.querySelector(".message-content");
                }

                // Append token to answer
                fullAnswer += data.content;
                // Use innerHTML instead of textContent to support HTML/math
                contentDiv.innerHTML = fullAnswer;

                // Auto-scroll
                const messagesArea = document.getElementById("messagesArea");
                messagesArea.scrollTop = messagesArea.scrollHeight;
              } else if (data.type === "sources") {
                sources = data.content;
                console.log("Sources:", sources);
              } else if (data.type === "error") {
                contentDiv.innerHTML = data.content;
                console.error("Stream error:", data.content);
              } else if (data.type === "done") {
                console.log("Stream done");
                streamComplete = true;

                // Trigger MathJax to render all math formulas
                if (typeof MathJax !== "undefined") {
                  MathJax.typesetPromise([contentDiv]).catch((err) => {
                    console.log("MathJax rendering error:", err);
                  });
                }

                // Add sources if available
                if (sources.length > 0) {
                  const sourcesDiv = document.createElement("div");
                  sourcesDiv.className = "message-sources";
                  // Format sources: only show filenames
                  const formattedSources = sources.map(s => {
                    if (typeof s === 'object' && s.filename) {
                      return s.filename;
                    }
                    return String(s);
                  }).join(", ");
                  sourcesDiv.innerHTML = `<strong>Sources:</strong> ${formattedSources}`;
                  contentDiv.appendChild(sourcesDiv);
                }
              }
            } catch (e) {
              console.error("JSON parse error:", e, "Line:", line);
            }
          }
        }
      }
    } catch (error) {
      console.error("Streaming error:", error);
      // Remove loading if still showing
      this.removeLoadingIndicator();
      // Create message container if not created
      if (!messageDiv) {
        messageDiv = this.createBotMessageContainer();
        contentDiv = messageDiv.querySelector(".message-content");
      }
      contentDiv.textContent =
        "Error: Failed to get response. Please try again.";
    } finally {
      // IMPORTANT: Always enable input when streaming completes or fails
      if (streamComplete) {
        this.enableInput();
      }
    }
  }

  createBotMessageContainer() {
    // Create a message container for bot response
    const messagesArea = document.getElementById("messagesArea");

    // Remove welcome message if exists
    const welcomeMsg = document.querySelector(".welcome-message");
    if (welcomeMsg) welcomeMsg.remove();

    // Create message element
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot";

    // Create avatar
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🤖";

    // Create content
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = "";

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesArea.appendChild(messageDiv);

    // Auto-scroll to bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;

    return messageDiv;
  }

  addMessage(sender, text, isBot, sources = []) {
    // Add a message to the chat display
    const messagesArea = document.getElementById("messagesArea");

    // Remove welcome message if exists
    const welcomeMsg = document.querySelector(".welcome-message");
    if (welcomeMsg) welcomeMsg.remove();

    // Create message element
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${isBot ? "bot" : "user"}`;

    // Create avatar
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = isBot ? "🤖" : "👤";

    // Create content
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    // Use innerHTML to allow HTML entities and preserve formatting
    // This enables proper rendering of <br>, entities, and math
    contentDiv.innerHTML = text;

    // Add sources if available
    if (sources && sources.length > 0) {
      const sourcesDiv = document.createElement("div");
      sourcesDiv.className = "message-sources";
      // Format sources: only show filenames
      const formattedSources = sources.map(s => {
        if (typeof s === 'object' && s.filename) {
          return s.filename;
        }
        return String(s);
      }).join(", ");
      sourcesDiv.innerHTML = `<strong>Sources:</strong> ${formattedSources}`;
      contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesArea.appendChild(messageDiv);

    // Trigger MathJax to render math in the new content
    if (isBot && typeof MathJax !== "undefined") {
      MathJax.typesetPromise([contentDiv]).catch((err) => {
        console.log("MathJax rendering error:", err);
      });
    }

    // Auto-scroll to bottom
    messagesArea.scrollTop = messagesArea.scrollHeight;
  }

  showLoadingIndicator() {
    // Display typing indicator while waiting for response
    const messagesArea = document.getElementById("messagesArea");
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "message bot";
    loadingDiv.id = "loading-indicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🤖";

    const contentDiv = document.createElement("div");
    contentDiv.className = "thinking-indicator";
    contentDiv.innerHTML = `
      <div class="thinking-content">
        <div class="thinking-spinner"></div>
        <span class="thinking-text">Thinking...</span>
      </div>
    `;

    loadingDiv.appendChild(avatar);
    loadingDiv.appendChild(contentDiv);
    messagesArea.appendChild(loadingDiv);
    messagesArea.scrollTop = messagesArea.scrollHeight;
  }

  removeLoadingIndicator() {
    // Remove the typing indicator
    const loading = document.getElementById("loading-indicator");
    if (loading) loading.remove();
  }

  async clearFiles() {
    // Display confirmation dialog with detailed warning
    const confirmed = confirm(
      "⚠️ Warning!\n\n" +
        "Are you sure you want to:\n" +
        "• Delete all uploaded files?\n" +
        "• Clear chat history?\n\n" +
        "This action cannot be undone.",
    );

    if (!confirmed) return;

    try {
      // Update status indicator to show deletion in progress
      this.updateStatus("Deleting...");
      const statusDiv = document.getElementById("uploadStatus");
      if (statusDiv) {
        statusDiv.innerHTML =
          '<div style="color: orange;">🗑️ Deleting files...</div>';
      }

      // Send DELETE request to backend API
      const response = await fetch(`${API_BASE_URL}/clear-files`, {
        method: "DELETE",
      });

      if (response.ok) {
        const data = await response.json();

        // Clear local files array
        this.files = [];

        // Reload uploaded files list (should be empty now)
        await this.loadUploadedFiles();

        // Reset chat area to welcome message
        const messagesArea = document.getElementById("messagesArea");
        messagesArea.innerHTML = `
    		    <div class="welcome-message">
                    <img src="logo.svg" alt="Viana" class="welcome-logo" onerror="this.outerHTML='<div class=\\'welcome-logo\\' style=\\'font-size: 80px;\\'>🤖</div>'">
                `;

        // Clear message input field
        const input = document.getElementById("messageInput");
        if (input) input.value = "";

        // Update status back to active
        this.updateStatus("Active");

        // Show success message
        if (statusDiv) {
          statusDiv.innerHTML =
            '<div style="color: green;">✅ All files deleted</div>';
          // Auto-hide after 3 seconds
          setTimeout(() => {
            statusDiv.innerHTML = "";
          }, 3000);
        }

        console.log("✅ Files cleared successfully");
      } else {
        // Parse error response from backend
        const error = await response.json();
        throw new Error(error.detail || error.message || "Failed to delete files");
      }
    } catch (error) {
      // Handle and display errors
      console.error("Error clearing files:", error);
      this.updateStatus("Error");

      const statusDiv = document.getElementById("uploadStatus");
      if (statusDiv) {
        statusDiv.innerHTML = `<div style="color: red;">❌ Error: ${error.message}</div>`;
      }

      alert("❌ Failed to delete files. Please try again.");
    }
  }

  async deleteFile(filename) {
    // Find the file item element and show loading state
    const filesList = document.getElementById("filesList");
    const fileItems = filesList.querySelectorAll(".file-item");
    let fileItem = null;

    fileItems.forEach((item) => {
      if (item.querySelector(".file-name").textContent.includes(filename)) {
        fileItem = item;
      }
    });

    // Add loading state to the file item
    if (fileItem) {
      fileItem.classList.add("deleting");
      const deleteBtn = fileItem.querySelector(".file-delete-btn");
      if (deleteBtn) {
        deleteBtn.disabled = true;
        deleteBtn.innerHTML = "⏳";
      }
    }

    this.updateStatus("Deleting...");

    try {
      const response = await fetch(
        `${API_BASE_URL}/delete-file/${encodeURIComponent(filename)}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        // Remove from local array
        this.files = this.files.filter((f) => f !== filename);
        // Refresh the list
        await this.loadUploadedFiles();
        this.updateStatus("Active");
        console.log(`✅ File deleted: ${filename}`);
      } else {
        const error = await response.json();
        throw new Error(error.detail || error.message || "Failed to delete file");
      }
    } catch (error) {
      console.error("Error deleting file:", error);
      this.updateStatus("Error");

      // Remove loading state on error
      if (fileItem) {
        fileItem.classList.remove("deleting");
        const deleteBtn = fileItem.querySelector(".file-delete-btn");
        if (deleteBtn) {
          deleteBtn.disabled = false;
          deleteBtn.innerHTML = "×";
        }
      }
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.novira = new NoviraChat();
  console.log("🚀 Viana Chat System initialized");
});
