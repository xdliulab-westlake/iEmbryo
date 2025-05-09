// code/step3_get_embryo_ids.js

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const https = require('https');

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/embryo_ids');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step3_get_embryo_ids.log');
const logFile = fs.createWriteStream(logFilePath, { flags: 'a' });

const originalConsoleLog = console.log;
const originalConsoleError = console.error;

console.log = function (message, ...optionalParams) {
    originalConsoleLog(message, ...optionalParams);
    logFile.write('[LOG] ' + message + '\n');
    if (optionalParams.length > 0) {
        logFile.write(JSON.stringify(optionalParams) + '\n');
    }
};

console.error = function (message, ...optionalParams) {
    originalConsoleError(message, ...optionalParams);
    logFile.write('[ERROR] ' + message + '\n');
    if (optionalParams.length > 0) {
        logFile.write(JSON.stringify(optionalParams) + '\n');
    }
};

// Set up HTTPS agent to ignore self-signed certificates
const agent = new https.Agent({
    rejectUnauthorized: false
});

// Read patients.json
const patientsFilePath = path.resolve(__dirname, '../output/patients.json');
let patientsData;
try {
    patientsData = JSON.parse(fs.readFileSync(patientsFilePath, 'utf8'));
} catch (err) {
    console.error('Failed to read patients.json:', err);
    process.exit(1);
}

const patients = patientsData.Patients;

// Declare authToken and authTokenTimestamp at the top-level scope
let authToken;
let authTokenTimestamp;

// Function to authenticate and get the token
async function authenticate() {
    try {
        const response = await axios.get('https://192.168.1.100:4000/login', {
            params: {
                Username: 'web',
                Password: 'web'
            },
            headers: {
                'Content-Type': 'application/json'
            },
            httpsAgent: agent
        });

        if (response.status === 200) {
            const responseJson = response.data;
            const token = responseJson.Token;
            return token;
        } else {
            console.error('Authentication failed with status code:', response.status);
            throw new Error('Authentication failed');
        }
    } catch (err) {
        console.error('Authentication request failed:', err);
        throw err;
    }
}

// Function to check if the token has expired
function checkTokenExpiration() {
    const currentTime = Date.now();
    if (currentTime - authTokenTimestamp >= 60 * 60 * 1000) { // 60 minutes
        console.log('Token expired. Re-authenticating...');
        return authenticate().then(token => {
            authToken = token;
            authTokenTimestamp = Date.now();
            console.log('Re-authentication successful.');
        }).catch(err => {
            console.error('Failed to re-authenticate:', err);
            throw err;
        });
    } else {
        return Promise.resolve();
    }
}

// Function to sanitize filenames
function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9]/gi, '_');
}

// Function to get embryo IDs for a treatment
async function getEmbryoIDs(patientIDx, treatmentName, retries = 0) {
    try {
        await checkTokenExpiration();

        const response = await axios.get('https://192.168.1.100:4000/GET/embryoID', {
            params: {
                PatientIDx: patientIDx,
                TreatmentName: treatmentName
            },
            headers: {
                'API-token': authToken,
                'Content-Type': 'application/json'
            },
            httpsAgent: agent,
            validateStatus: function (status) {
                return true; // Accept all status codes
            }
        });

        if (response.status === 200) {
            const responseBody = response.data;
            // Sanitize file name to avoid illegal characters
            const sanitizedPatientIDx = sanitizeFilename(patientIDx);
            const sanitizedTreatmentName = sanitizeFilename(treatmentName);
            const filePath = path.join(outputDir, `embryo_ids_${sanitizedPatientIDx}_${sanitizedTreatmentName}.json`);
            fs.writeFileSync(filePath, JSON.stringify(responseBody));
            console.log(`Embryo IDs for PatientIDx ${patientIDx}, TreatmentName ${treatmentName} saved successfully.`);
        } else if (response.status === 401) {
            console.warn(`Unauthorized access for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}. Attempting to re-authenticate...`);

            // Re-authenticate and retry the request
            if (retries < 3) {
                try {
                    authToken = await authenticate();
                    authTokenTimestamp = Date.now();
                    console.log('Re-authentication successful. Retrying request...');
                    await getEmbryoIDs(patientIDx, treatmentName, retries + 1);
                } catch (authErr) {
                    console.error('Failed to re-authenticate:', authErr);
                }
            } else {
                console.error(`Maximum retries reached for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}.`);
            }
        } else if (response.status === 204) {
            console.log(`No embryo IDs found for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}.`);
        } else if (response.status === 429) {
            console.error(`Too many requests for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}: Status code ${response.status}`);
            // Wait and retry
            if (retries < 3) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                await getEmbryoIDs(patientIDx, treatmentName, retries + 1);
            } else {
                console.error(`Maximum retries reached for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}.`);
            }
        } else {
            console.error(`Error retrieving embryo IDs for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}: Status code ${response.status}`);
        }
    } catch (err) {
        console.error(`Request failed for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}:`, err);
    }
}

// Main function
(async () => {
    try {
        authToken = await authenticate();
        authTokenTimestamp = Date.now(); // Record the time the token was acquired
        console.log('Authentication successful.');
    } catch (err) {
        console.error('Failed to authenticate:', err);
        process.exit(1);
    }

    // Iterate over patients and their treatments
    for (const patient of patients) {
        const patientIDx = patient.PatientIDx;
        console.log(`Processing PatientIDx: ${patientIDx}`);

        // Read treatments for this patient
        const treatmentsFilePath = path.resolve(__dirname, `../output/treatments/treatments_${sanitizeFilename(patientIDx)}.json`);
        let treatmentsData;
        try {
            treatmentsData = JSON.parse(fs.readFileSync(treatmentsFilePath, 'utf8'));
        } catch (err) {
            console.error(`Failed to read treatments for PatientIDx ${patientIDx}:`, err);
            continue;
        }

        const treatments = treatmentsData.TreatmentList;

        if (!treatments || treatments.length === 0) {
            console.log(`No treatments found for PatientIDx ${patientIDx}.`);
            continue;
        }

        // Iterate over treatments
        for (const treatmentName of treatments) {
            console.log(`  Processing TreatmentName: ${treatmentName}`);

            try {
                await getEmbryoIDs(patientIDx, treatmentName);
            } catch (err) {
                console.error(`Error processing PatientIDx ${patientIDx}, TreatmentName ${treatmentName}:`, err);
            }

            // Wait 100ms between requests to comply with API rate limit
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    console.log('All embryo IDs retrieved.');

    // Close the log file
    logFile.end();
})();