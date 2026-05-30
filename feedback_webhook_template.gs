/**
 * Google Apps Script — White RAG Investor Feedback Webhook
 *
 * SETUP:
 * 1. Go to https://script.google.com and create a new project.
 * 2. Paste this entire script into the editor.
 * 3. Replace SPREADSHEET_ID with your Google Sheet ID
 *    (the long string in the Sheet URL between /d/ and /edit).
 * 4. Click Deploy > New deployment > Web app.
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 5. Copy the deployment URL.
 * 6. Add it to your Streamlit secrets as:
 *    FEEDBACK_WEBHOOK_URL = "https://script.google.com/macros/s/YOUR_ID/exec"
 *
 * The Sheet should have these column headers in Row 1:
 *   A: Timestamp | B: Feedback | C: Answer Preview | D: Message Index
 */

const SPREADSHEET_ID = "YOUR_SPREADSHEET_ID_HERE";
const SHEET_NAME = "Feedback";

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName(SHEET_NAME);

    if (!sheet) {
      // Create the sheet if it doesn't exist
      const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
      const newSheet = ss.insertSheet(SHEET_NAME);
      newSheet.appendRow(["Timestamp", "Feedback", "Answer Preview", "Message Index"]);
      newSheet.appendRow([
        data.timestamp || "",
        data.feedback || "",
        data.answer_preview || "",
        data.message_index || "",
      ]);
    } else {
      sheet.appendRow([
        data.timestamp || "",
        data.feedback || "",
        data.answer_preview || "",
        data.message_index || "",
      ]);
    }

    return ContentService.createTextOutput(
      JSON.stringify({ status: "ok" })
    ).setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(
      JSON.stringify({ status: "error", message: err.toString() })
    ).setMimeType(ContentService.MimeType.JSON);
  }
}
