// Global error handler to display runtime errors visually for easier debugging
window.addEventListener('error', (event) => {
    console.error("Captured unhandled error:", event.error);
    const errBox = document.createElement('div');
    errBox.className = 'debug-error-box';
    errBox.style.position = 'fixed';
    errBox.style.bottom = '10px';
    errBox.style.left = '10px';
    errBox.style.right = '10px';
    errBox.style.background = '#f87171';
    errBox.style.color = '#fff';
    errBox.style.padding = '12px';
    errBox.style.borderRadius = '8px';
    errBox.style.zIndex = '99999';
    errBox.style.fontSize = '12px';
    errBox.style.fontFamily = 'monospace';
    errBox.style.boxShadow = '0 4px 20px rgba(0,0,0,0.4)';
    errBox.style.border = '1px solid rgba(255,255,255,0.2)';
    errBox.innerHTML = `<strong>JavaScript Error:</strong> ${event.message} <br><span style="opacity:0.8;">at ${event.filename.split('/').pop()}:${event.lineno}:${event.colno}</span>`;
    document.body.appendChild(errBox);
});

// Safe storage helper to prevent SecurityError crashes in private browsing/strict privacy modes (e.g. Firefox)
const safeStorage = {
    getItem(key) {
        try {
            return localStorage.getItem(key);
        } catch (e) {
            console.warn('Storage access blocked:', e);
            return null;
        }
    },
    setItem(key, value) {
        try {
            localStorage.setItem(key, value);
        } catch (e) {
            console.warn('Storage write blocked:', e);
        }
    },
    removeItem(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('Storage remove blocked:', e);
        }
    }
};

// State Management
let chatHistory = [];
let questionCount = 0;
const MAX_QUESTIONS = 25;

// Safe markdown parsing wrapper
function parseMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text);
    }
    // Fallback basic text formatter if CDN fails
    console.warn('Marked library not loaded. Using fallback plain text formatter.');
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
        .replace(/\n/g, '<br>');
}

// DOM Elements
let sidebar;
let sidebarOverlay;
let sidebarToggle;
let btnNewChat;
let btnExportChat;
let btnCopySummary;
let selectResponseMode;

// Calculator Elements
let calcHeader;
let calcBody;
let calcToggleIcon;
let calcLoanBalance;
let calcLoanRate;
let calcInvReturn;
let calcExtraPmt;
let calcResults;

// Chat Elements
let chatMessages;
let chatForm;
let chatInput;
let startersContainer;
let startersGrid;

function initDOMElements() {
    sidebar = document.getElementById('sidebar');
    sidebarOverlay = document.getElementById('sidebar-overlay');
    sidebarToggle = document.getElementById('sidebar-toggle');
    btnNewChat = document.getElementById('btn-new-chat');
    btnExportChat = document.getElementById('btn-export-chat');
    btnCopySummary = document.getElementById('btn-copy-summary');
    selectResponseMode = document.getElementById('select-response-mode');

    calcHeader = document.getElementById('calc-header');
    calcBody = document.getElementById('calc-body');
    calcToggleIcon = document.getElementById('calc-toggle-icon');
    calcLoanBalance = document.getElementById('calc-loan-balance');
    calcLoanRate = document.getElementById('calc-loan-rate');
    calcInvReturn = document.getElementById('calc-inv-return');
    calcExtraPmt = document.getElementById('calc-extra-pmt');
    calcResults = document.getElementById('calc-results');

    chatMessages = document.getElementById('chat-messages');
    chatForm = document.getElementById('chat-form');
    chatInput = document.getElementById('chat-input');
    startersContainer = document.getElementById('conversation-starters');
    startersGrid = document.getElementById('starters-grid');
}

// PWA Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(reg => console.log('Service Worker registered.'))
            .catch(err => console.log('Service Worker registration failed:', err));
    });
}

// App Initialization
document.addEventListener('DOMContentLoaded', () => {
    try {
        initDOMElements();
        
        // Clear chat history and question limit count on page reload to start fresh
        safeStorage.removeItem('wri_chat_history');
        safeStorage.removeItem('wri_question_count');
        
        loadChatHistory();
        initEventListeners();
        fetchStarters();
        runCalculator(); // Run initial calc
    } catch (e) {
        console.error("Initialization error:", e);
        throw e; // Rethrow to let global window error handler capture it visually
    }
});

// Event Listeners
function initEventListeners() {
    // Mobile Sidebar Toggle
    sidebarToggle.addEventListener('click', toggleSidebar);
    sidebarOverlay.addEventListener('click', toggleSidebar);

    // Chat Actions
    btnNewChat.addEventListener('click', resetChat);
    btnExportChat.addEventListener('click', exportChat);
    btnCopySummary.addEventListener('click', copySummary);

    // Select input change
    selectResponseMode.addEventListener('change', () => {
        safeStorage.setItem('wri_response_mode', selectResponseMode.value);
    });
    // Load cached response mode
    const cachedMode = safeStorage.getItem('wri_response_mode');
    if (cachedMode) {
        selectResponseMode.value = cachedMode;
    }

    // Calculator Toggle Expand/Collapse
    calcHeader.addEventListener('click', () => {
        calcBody.classList.toggle('expanded');
        calcToggleIcon.classList.toggle('rotated');
    });

    // Calculator Inputs Live Update (Safely bind to both input and change events)
    const calcInputs = [
        document.getElementById('calc-loan-balance'),
        document.getElementById('calc-loan-rate'),
        document.getElementById('calc-inv-return'),
        document.getElementById('calc-extra-pmt')
    ];
    calcInputs.forEach(input => {
        if (input) {
            input.addEventListener('input', runCalculator);
            input.addEventListener('change', runCalculator);
        }
    });

    // Chat Input Self-Resizing & Enter key submit
    chatInput.addEventListener('input', () => {
        chatInput.style.height = '24px';
        const scrollHeight = chatInput.scrollHeight;
        if (scrollHeight > 24) {
            chatInput.style.height = scrollHeight + 'px';
        }
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Chat Submit
    chatForm.addEventListener('submit', handleFormSubmit);

    // Disclaimer Close Event
    const disclaimerBanner = document.getElementById('disclaimer-banner');
    const btnCloseDisclaimer = document.getElementById('btn-close-disclaimer');
    if (disclaimerBanner && btnCloseDisclaimer) {
        btnCloseDisclaimer.addEventListener('click', () => {
            disclaimerBanner.style.display = 'none';
            safeStorage.setItem('wri_disclaimer_dismissed', 'true');
        });

        // Load disclaimer dismissed state
        if (safeStorage.getItem('wri_disclaimer_dismissed') === 'true') {
            disclaimerBanner.style.display = 'none';
        }
    }
}

// Sidebar Drawer Toggle for Mobile
function toggleSidebar() {
    sidebar.classList.toggle('active');
    sidebarOverlay.classList.toggle('active');
}

function openSidebar() {
    sidebar.classList.add('active');
    sidebarOverlay.classList.add('active');
}

// Load Chat History from LocalStorage
function loadChatHistory() {
    const savedHistory = safeStorage.getItem('wri_chat_history');
    const savedCount = safeStorage.getItem('wri_question_count');
    
    questionCount = savedCount ? parseInt(savedCount, 10) : 0;
    
    if (savedHistory) {
        chatHistory = JSON.parse(savedHistory);
        if (chatHistory.length > 0) {
            const hasUserMessages = chatHistory.some(msg => msg.role === 'user');
            if (hasUserMessages) {
                startersContainer.style.display = 'none';
            } else {
                startersContainer.style.display = 'block';
            }
            chatHistory.forEach((msg, idx) => {
                renderMessageBubble(msg.role, msg.content, {
                    confidence: msg.confidence,
                    confidenceText: msg.confidence_text,
                    sources: msg.sources,
                    followUps: msg.followUps,
                    feedback: msg.feedback,
                    msgIndex: idx,
                    isWelcome: idx === 0
                });
            });
            scrollToBottom();
            return;
        }
    }
    
    // Set default greeting if history is empty
    const welcomeMsg = {
        role: 'assistant',
        content: "Welcome to **White RAG Investor** — a financial advisor trained on the entire White Coat Investor blog.\n\nAsk me anything about physician finances: student loans, disability insurance, investing, contracts, taxes, and more."
    };
    chatHistory = [welcomeMsg];
    renderMessageBubble(welcomeMsg.role, welcomeMsg.content, { isWelcome: true });
    saveChatHistory();
}

// Save Chat History
function saveChatHistory() {
    safeStorage.setItem('wri_chat_history', JSON.stringify(chatHistory));
    safeStorage.setItem('wri_question_count', questionCount.toString());
}

// Reset Chat (New Chat)
function resetChat() {
    if (confirm('Are you sure you want to start a new chat? This will clear your current conversation.')) {
        chatHistory = [];
        questionCount = 0;
        safeStorage.removeItem('wri_chat_history');
        safeStorage.removeItem('wri_question_count');
        chatMessages.innerHTML = '';
        startersContainer.style.display = 'block';
        loadChatHistory();
        fetchStarters();
        // Close sidebar on mobile
        if (sidebar.classList.contains('active')) {
            toggleSidebar();
        }
    }
}

// Fetch Dynamic Conversation Starters
async function fetchStarters() {
    try {
        const response = await fetch('/api/starters');
        const data = await response.json();
        renderStarters(data.starters);
    } catch (err) {
        console.error('Failed to load starters:', err);
        renderStarters([
            "Disability insurance basics",
            "Should I refinance my student loans?",
            "How to start investing as a resident",
            "Backdoor Roth IRA explained"
        ]);
    }
}

function renderStarters(starters) {
    startersGrid.innerHTML = '';
    starters.forEach(starter => {
        const btn = document.createElement('button');
        btn.className = 'btn-starter';
        btn.textContent = starter;
        btn.addEventListener('click', () => {
            sendUserQuery(starter);
        });
        startersGrid.appendChild(btn);
    });
}

// Run Loan vs Investing Calculator Math
function runCalculator() {
    try {
        const calcLoanBalance = document.getElementById('calc-loan-balance');
        const calcLoanRate = document.getElementById('calc-loan-rate');
        const calcInvReturn = document.getElementById('calc-inv-return');
        const calcExtraPmt = document.getElementById('calc-extra-pmt');
        const calcResults = document.getElementById('calc-results');

        if (!calcLoanBalance || !calcLoanRate || !calcInvReturn || !calcExtraPmt || !calcResults) {
            console.warn("Calculator DOM elements not yet initialized.");
            return;
        }

        const balance = parseFloat(calcLoanBalance.value) || 0;
        const rate = (parseFloat(calcLoanRate.value) || 0) / 100.0;
        const returnRate = (parseFloat(calcInvReturn.value) || 0) / 100.0;
        const extra = parseFloat(calcExtraPmt.value) || 0;

        if (extra <= 0 || balance <= 0) {
            calcResults.innerHTML = '<div class="calc-badge error">Please enter a valid loan balance and extra monthly cash.</div>';
            return;
        }

        const r_m = rate / 12.0;
        let monthsToPay = 0;
        
        if (r_m === 0) {
            monthsToPay = balance / extra;
        } else {
            const formulaVal = 1.0 - (balance * r_m) / extra;
            if (formulaVal > 0) {
                monthsToPay = -Math.log(formulaVal) / Math.log(1.0 + r_m);
            } else {
                monthsToPay = Infinity;
            }
        }

        if (monthsToPay !== Infinity && !isNaN(monthsToPay)) {
            const yearsToPay = monthsToPay / 12.0;
            const totalPaid = extra * monthsToPay;
            const interestPaid = totalPaid - balance;

            const r_inv_m = returnRate / 12.0;
            let investedVal = 0;
            
            if (r_inv_m === 0) {
                investedVal = extra * monthsToPay;
            } else {
                investedVal = extra * ((Math.pow(1.0 + r_inv_m, monthsToPay) - 1.0) / r_inv_m);
            }

            const earnings = investedVal - totalPaid;
            const loanFutureValue = balance * Math.pow(1.0 + r_m, monthsToPay);
            const netDiff = investedVal - loanFutureValue;

            // Format numbers to match Streamlit formatting
            const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
            
            let badgeHtml = '';
            if (netDiff > 0) {
                badgeHtml = `<div class="calc-badge success">Investing wins by: <strong>$${formatter.format(netDiff)}</strong></div>`;
            } else {
                badgeHtml = `<div class="calc-badge info">Paying off loan wins by: <strong>$${formatter.format(-netDiff)}</strong></div>`;
            }

            calcResults.innerHTML = `
                <p>Payoff Period: <strong>${yearsToPay.toFixed(1)} years</strong></p>
                <p>Total Interest Paid: <strong>$${formatter.format(interestPaid)}</strong></p>
                <p>Alternative Investment Value: <strong>$${formatter.format(investedVal)}</strong> <span style="font-size:0.75rem;color:var(--text-secondary);">(Growth: $${formatter.format(earnings)})</span></p>
                ${badgeHtml}
            `;
        } else {
            calcResults.innerHTML = '<div class="calc-badge error">Extra monthly cash is too low to cover the monthly interest.</div>';
        }
    } catch (e) {
        console.error("Calculator logic error:", e);
        const resultsEl = document.getElementById('calc-results');
        if (resultsEl) {
            resultsEl.innerHTML = `<div class="calc-badge error">Error: ${e.message}</div>`;
        }
    }
}

// Export Chat Transcript
function exportChat() {
    if (chatHistory.length <= 1) {
        alert('No conversation to export.');
        return;
    }
    
    let text = "WHITE RAG INVESTOR - CHAT TRANSCRIPT\n";
    text += "========================================\n\n";
    
    chatHistory.forEach(msg => {
        const role = msg.role === 'user' ? 'User' : 'Assistant';
        text += `${role}:\n${msg.content}\n\n`;
        if (msg.confidence) {
            text += `[Confidence: ${msg.confidence_text}]\n`;
        }
        if (msg.sources && msg.sources.length > 0) {
            text += "[Sources Referenced]:\n";
            msg.sources.forEach(src => {
                text += `- [${src.id}] ${src.title} (${src.url})\n`;
            });
        }
        text += "----------------------------------------\n\n";
    });
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `white_rag_investor_chat_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// Copy Summary of Chat to Clipboard
function copySummary() {
    if (chatHistory.length <= 1) {
        alert('No conversation to summarize.');
        return;
    }
    
    // Build a nice clean markdown summary
    const summaryLines = ["### Chat Summary - White RAG Investor"];
    
    chatHistory.forEach(msg => {
        if (msg.role === 'user') {
            summaryLines.push(`**Q:** ${msg.content}`);
        } else if (msg.role === 'assistant' && msg.sources) {
            // Include just a snippet of assistant response and source counts
            const intro = msg.content.substring(0, 200).replace(/\n/g, ' ') + '...';
            summaryLines.push(`**A:** ${intro}`);
            const uniqueUrls = [...new Set(msg.sources.map(s => s.url))];
            summaryLines.push(`*References:* ${uniqueUrls.length} WCI article(s) cited.`);
            summaryLines.push("");
        }
    });

    navigator.clipboard.writeText(summaryLines.join('\n')).then(() => {
        const originalText = btnCopySummary.innerHTML;
        btnCopySummary.innerHTML = `
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="var(--semantic-success)" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            <span style="color:var(--semantic-success)">Copied!</span>
        `;
        setTimeout(() => {
            btnCopySummary.innerHTML = originalText;
        }, 2000);
    }).catch(err => {
        alert('Failed to copy summary to clipboard.');
    });
}

// Append a message bubble to the canvas container
function renderMessageBubble(role, content, options = {}) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble message-${role}`;
    
    const roleLabel = document.createElement('div');
    roleLabel.className = 'message-role-label';
    roleLabel.textContent = role === 'user' ? 'User' : 'WRI Advisor';
    bubble.appendChild(roleLabel);
    
    const textNode = document.createElement('div');
    textNode.className = 'markdown-content';
    textNode.innerHTML = parseMarkdown(content);
    bubble.appendChild(textNode);
    
    // Add confidence tag (if assistant message and on-topic)
    if (role === 'assistant' && options.confidence && options.confidence !== 'none') {
        const confBadge = document.createElement('span');
        confBadge.className = `confidence-tag confidence-${options.confidence}`;
        confBadge.textContent = options.confidenceText || options.confidence;
        bubble.appendChild(confBadge);
    }
    
    // Add expandable details for sources
    if (role === 'assistant' && options.sources && options.sources.length > 0) {
        const details = document.createElement('details');
        details.className = 'sources-details';
        
        const summary = document.createElement('summary');
        summary.className = 'sources-summary';
        const numArticles = [...new Set(options.sources.map(s => s.url))].length;
        summary.textContent = `View sources (${numArticles} article${numArticles !== 1 ? 's' : ''} referenced)`;
        details.appendChild(summary);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'sources-content';
        
        options.sources.forEach(src => {
            const item = document.createElement('div');
            item.className = 'source-item';
            
            const titleLine = document.createElement('div');
            titleLine.className = 'source-item-title';
            const yearText = src.year ? ` (${src.year})` : '';
            titleLine.innerHTML = `Excerpt [${src.id}]: <a href="${src.url}" target="_blank">${src.title}</a>${yearText}`;
            item.appendChild(titleLine);
            
            if (src.content) {
                const bodyLine = document.createElement('div');
                bodyLine.className = 'source-item-body';
                bodyLine.textContent = src.content;
                item.appendChild(bodyLine);
            }
            contentDiv.appendChild(item);
        });
        
        details.appendChild(contentDiv);
        bubble.appendChild(details);
    }
    
    // Add thumbs feedback widget
    if (role === 'assistant' && options.msgIndex !== undefined && options.msgIndex > 0) {
        const feedbackVal = options.feedback; // undefined, 'up', or 'down'
        
        const feedbackWidget = document.createElement('div');
        feedbackWidget.className = 'feedback-widget';
        
        const btnUp = document.createElement('button');
        btnUp.className = `btn-feedback ${feedbackVal === 'up' ? 'active' : ''}`;
        btnUp.setAttribute('aria-label', 'Thumbs Up');
        btnUp.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>`;
        
        const btnDown = document.createElement('button');
        btnDown.className = `btn-feedback ${feedbackVal === 'down' ? 'active' : ''}`;
        btnDown.setAttribute('aria-label', 'Thumbs Down');
        btnDown.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm8-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>`;
        
        const statusSpan = document.createElement('span');
        statusSpan.className = 'feedback-status';
        if (feedbackVal) statusSpan.textContent = 'Feedback saved';
        
        const handleFeedbackClick = async (valStr, isUpVal) => {
            if (btnUp.disabled || feedbackVal !== undefined) return;
            
            btnUp.disabled = true;
            btnDown.disabled = true;
            statusSpan.textContent = 'Submitting...';
            
            try {
                const fVal = isUpVal ? 1 : 0;
                const req = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        feedback_value: fVal,
                        message_content: content,
                        message_index: options.msgIndex
                    })
                });
                if (req.ok) {
                    statusSpan.textContent = 'Feedback sent!';
                    if (isUpVal) btnUp.classList.add('active');
                    else btnDown.classList.add('active');
                    
                    // Save feedback in state & localStorage
                    chatHistory[options.msgIndex].feedback = valStr;
                    saveChatHistory();
                } else {
                    statusSpan.textContent = 'Failed to send';
                    btnUp.disabled = false;
                    btnDown.disabled = false;
                }
            } catch (err) {
                statusSpan.textContent = 'Network error';
                btnUp.disabled = false;
                btnDown.disabled = false;
            }
        };

        if (feedbackVal === undefined) {
            btnUp.addEventListener('click', () => handleFeedbackClick('up', true));
            btnDown.addEventListener('click', () => handleFeedbackClick('down', false));
        } else {
            btnUp.disabled = true;
            btnDown.disabled = true;
        }

        feedbackWidget.appendChild(btnUp);
        feedbackWidget.appendChild(btnDown);
        feedbackWidget.appendChild(statusSpan);
        bubble.appendChild(feedbackWidget);
    }
    
    // Add suggested follow-up questions
    if (role === 'assistant' && options.followUps && options.followUps.length > 0) {
        const followupsDiv = document.createElement('div');
        followupsDiv.className = 'followups-container';
        
        const title = document.createElement('div');
        title.className = 'followups-title';
        title.textContent = 'Suggested follow-up questions:';
        followupsDiv.appendChild(title);
        
        options.followUps.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'btn-followup';
            btn.textContent = q;
            btn.addEventListener('click', () => {
                sendUserQuery(q);
            });
            followupsDiv.appendChild(btn);
        });
        bubble.appendChild(followupsDiv);
    }

    // Special Calculator CTA for welcome greeting
    if (role === 'assistant' && options.isWelcome) {
        const welcomeCalcDiv = document.createElement('div');
        welcomeCalcDiv.className = 'welcome-calc-container';
        welcomeCalcDiv.style.marginTop = '1.25rem';
        
        const welcomeCalcBtn = document.createElement('button');
        welcomeCalcBtn.className = 'btn btn-primary';
        welcomeCalcBtn.style.width = 'auto';
        welcomeCalcBtn.style.padding = '0.65rem 1.25rem';
        welcomeCalcBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="margin-right:6px;vertical-align:middle;">
                <rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect>
                <line x1="9" y1="22" x2="9" y2="16"></line>
                <line x1="8" y1="6" x2="16" y2="6"></line>
                <line x1="16" y1="14" x2="16" y2="22"></line>
                <line x1="9" y1="14" x2="15" y2="14"></line>
                <line x1="9" y1="10" x2="15" y2="10"></line>
            </svg>
            <span>Open Loan vs. Investing Calculator</span>
        `;
        welcomeCalcBtn.addEventListener('click', openSidebar);
        welcomeCalcDiv.appendChild(welcomeCalcBtn);
        bubble.appendChild(welcomeCalcDiv);
    }

    chatMessages.appendChild(bubble);
}

// Scroll chat panel to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Direct starter buttons click triggering messaging sequence
function sendUserQuery(text) {
    chatInput.value = text;
    chatForm.dispatchEvent(new Event('submit'));
}

// Form Submit Handler (Sends Message and decodes Server-Sent Events)
async function handleFormSubmit(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    if (questionCount >= MAX_QUESTIONS) {
        alert("You've reached the 25-question limit for this session. Please start a 'New Chat' using the button in the sidebar to reset.");
        return;
    }

    // 1. Hide Starters grid
    startersContainer.style.display = 'none';

    // 2. Add message to local history and render user bubble
    chatHistory.push({ role: 'user', content: text });
    renderMessageBubble('user', text);
    
    // Clear & Resize input
    chatInput.value = '';
    chatInput.style.height = '24px';
    scrollToBottom();

    // 3. Render loading status spinner
    const statusDiv = document.createElement('div');
    statusDiv.className = 'retrieval-status';
    statusDiv.innerHTML = `<div class="spinner"></div><span id="status-text">Searching knowledge base...</span>`;
    chatMessages.appendChild(statusDiv);
    scrollToBottom();

    const statusText = document.getElementById('status-text');

    // 4. Perform API fetch for stream response
    try {
        const mode = selectResponseMode.value;
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: chatHistory,
                response_mode: mode
            })
        });

        if (!response.ok) {
            throw new Error(`Server returned code ${response.status}`);
        }

        // Remove loading status spinner
        statusDiv.remove();

        // 5. Prepare target assistant message bubble for streaming
        const bubbleIndex = chatHistory.length;
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble message-assistant';
        
        const roleLabel = document.createElement('div');
        roleLabel.className = 'message-role-label';
        roleLabel.textContent = 'WRI Advisor';
        bubble.appendChild(roleLabel);
        
        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown-content streaming-cursor';
        bubble.appendChild(markdownDiv);
        
        chatMessages.appendChild(bubble);
        scrollToBottom();

        // 6. Decode the stream chunk by chunk
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let fullResponseText = '';
        let metadata = null;
        let followUps = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            // Process buffer line by line
            const lines = buffer.split('\n');
            // Keep the last partial line in buffer
            buffer = lines.pop();

            let currentEvent = 'message';
            for (let line of lines) {
                line = line.trim();
                if (!line) continue;

                if (line.startsWith('event:')) {
                    currentEvent = line.substring(6).trim();
                } else if (line.startsWith('data:')) {
                    const dataStr = line.substring(5).trim();
                    
                    if (currentEvent === 'metadata') {
                        metadata = JSON.parse(dataStr);
                        // Add confidence badge if available
                        if (metadata.confidence && metadata.confidence !== 'none') {
                            const confBadge = document.createElement('span');
                            confBadge.className = `confidence-tag confidence-${metadata.confidence}`;
                            confBadge.textContent = metadata.confidence_text || metadata.confidence;
                            bubble.appendChild(confBadge);
                        }
                    } else if (currentEvent === 'token') {
                        const token = JSON.parse(dataStr);
                        fullResponseText += token;
                        markdownDiv.innerHTML = parseMarkdown(fullResponseText);
                        scrollToBottom();
                    } else if (currentEvent === 'follow_ups') {
                        followUps = JSON.parse(dataStr);
                    }
                }
            }
        }

        // Clean up cursor animations
        markdownDiv.classList.remove('streaming-cursor');

        // Extract raw text before follow-up questions
        // In backend, we extract followups at done, but since LLM sends everything, 
        // the client's rendered markdown might contain follow-up text. We clean it:
        const pattern = /(You might also want to ask:|Recommended follow-up questions:)/i;
        const match = fullResponseText.search(pattern);
        let cleanedText = fullResponseText;
        if (match !== -1) {
            cleanedText = fullResponseText.substring(0, match).trim();
            // Update bubble HTML with cleaned text
            markdownDiv.innerHTML = parseMarkdown(cleanedText);
        }

        // Increment question count
        questionCount++;

        // Add sources expander if available in metadata
        if (metadata && metadata.sources && metadata.sources.length > 0) {
            const details = document.createElement('details');
            details.className = 'sources-details';
            
            const summary = document.createElement('summary');
            summary.className = 'sources-summary';
            const numArticles = [...new Set(metadata.sources.map(s => s.url))].length;
            summary.textContent = `View sources (${numArticles} article${numArticles !== 1 ? 's' : ''} referenced)`;
            details.appendChild(summary);
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'sources-content';
            
            metadata.sources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                
                const titleLine = document.createElement('div');
                titleLine.className = 'source-item-title';
                const yearText = src.year ? ` (${src.year})` : '';
                titleLine.innerHTML = `Excerpt [${src.id}]: <a href="${src.url}" target="_blank">${src.title}</a>${yearText}`;
                item.appendChild(titleLine);
                
                if (src.content) {
                    const bodyLine = document.createElement('div');
                    bodyLine.className = 'source-item-body';
                    bodyLine.textContent = src.content;
                    item.appendChild(bodyLine);
                }
                contentDiv.appendChild(item);
            });
            
            details.appendChild(contentDiv);
            bubble.appendChild(details);
        }

        // Save state of current assistant response inside array
        const messageObject = {
            role: 'assistant',
            content: cleanedText,
            confidence: metadata ? metadata.confidence : 'none',
            confidence_text: metadata ? metadata.confidence_text : '',
            sources: metadata ? metadata.sources : [],
            followUps: followUps,
            feedback: undefined
        };
        
        chatHistory.push(messageObject);
        saveChatHistory();

        // Rerender message bubble container fully with thumbs buttons and follow-up buttons
        // Remove the temporary bubble and render a permanent one to establish bindings
        bubble.remove();
        renderMessageBubble('assistant', cleanedText, {
            confidence: messageObject.confidence,
            confidenceText: messageObject.confidence_text,
            sources: messageObject.sources,
            followUps: messageObject.followUps,
            feedback: undefined,
            msgIndex: bubbleIndex
        });
        
        scrollToBottom();

    } catch (err) {
        console.error('Error fetching stream response:', err);
        statusDiv.remove();
        renderMessageBubble('assistant', `*Failed to connect to assistant: ${err.message}. Please check your connection or LLM status.*`);
        scrollToBottom();
    }
}
