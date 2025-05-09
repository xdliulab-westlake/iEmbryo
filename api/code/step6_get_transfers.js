// code/step5_get_transfers.js

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const https = require('https');

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/transfers');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step5_get_transfers.log');
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
    rejectUnauthorized: false,
    keepAlive: true,
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

// Function to authenticate and get the token
async function authenticate() {
    try {
        const response = await axios.get('https://192.168.1.202:4000/login', {
            params: {
                Username: 'web',
                Password: 'web'
            },
            headers: {
                'Content-Type': 'application/json'
            },
            httpsAgent: agent,
            timeout: 10000, // Set a timeout for authentication
        });

        if (response.status === 200) {
            const responseJson = response.data;
            const authToken = responseJson.Token;
            return authToken;
        } else {
            console.error('Authentication failed with status code:', response.status);
            throw new Error('Authentication failed');
        }
    } catch (err) {
        console.error('Authentication request failed:', err.message);
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

// Function to get transfers for a patient
async function getTransfers(patientIDx) {
    try {
        await checkTokenExpiration();

        const response = await axios.get('https://192.168.1.202:4000/GET/transfers', {
            params: {
                PatientIDx: patientIDx
            },
            headers: {
                'API-token': authToken,
                'Content-Type': 'application/json'
            },
            httpsAgent: agent,
            timeout: 60000, // Increased timeout duration
            validateStatus: function (status) {
                return true; // Accept all status codes
            },
        });

        if (response.status === 200) {
            const responseBody = response.data;
            const filePath = path.join(outputDir, `transfers_${sanitizeFilename(patientIDx)}.json`);
            fs.writeFileSync(filePath, JSON.stringify(responseBody, null, 2));
            console.log(`Transfers for PatientIDx ${patientIDx} saved successfully.`);
            return responseBody;
        } else if (response.status === 204) {
            console.log(`No transfers found for PatientIDx ${patientIDx}.`);
            return null;
        } else if (response.status === 401) {
            console.warn(`Unauthorized access for PatientIDx ${patientIDx}. Attempting to re-authenticate...`);
            await authenticate();
            authTokenTimestamp = Date.now();
            console.log('Re-authentication successful. Retrying request...');
            return await getTransfers(patientIDx);
        } else {
            console.error(`Error retrieving transfers for PatientIDx ${patientIDx}: Status code ${response.status}`);
            return null;
        }
    } catch (err) {
        console.error(`Request failed for PatientIDx ${patientIDx}:`, err.message);
        return null;
    }
}

// Function to get transfer info
async function getTransferInfo(transferID) {
    try {
        await checkTokenExpiration();

        const response = await axios.get('https://192.168.1.202:4000/GET/transferinfo', {
            params: {
                TransferID: transferID
            },
            headers: {
                'API-token': authToken,
                'Content-Type': 'application/json'
            },
            httpsAgent: agent,
            timeout: 60000, // Increased timeout duration
            validateStatus: function (status) {
                return true; // Accept all status codes
            },
        });

        if (response.status === 200) {
            const responseBody = response.data;
            const filePath = path.join(outputDir, `transferinfo_${sanitizeFilename(transferID)}.json`);
            fs.writeFileSync(filePath, JSON.stringify(responseBody, null, 2));
            console.log(`Transfer info for TransferID ${transferID} saved successfully.`);
        } else if (response.status === 204) {
            console.log(`No transfer info found for TransferID ${transferID}.`);
        } else if (response.status === 401) {
            console.warn(`Unauthorized access for TransferID ${transferID}. Attempting to re-authenticate...`);
            await authenticate();
            authTokenTimestamp = Date.now();
            console.log('Re-authentication successful. Retrying request...');
            await getTransferInfo(transferID);
        } else {
            console.error(`Error retrieving transfer info for TransferID ${transferID}: Status code ${response.status}`);
        }
    } catch (err) {
        console.error(`Request failed for TransferID ${transferID}:`, err.message);
    }
}

// Function to sanitize filenames
function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9]/gi, '_');
}

// Declare authToken and authTokenTimestamp at the top-level scope
let authToken;
let authTokenTimestamp;

// Main function
(async () => {
    try {
        authToken = await authenticate();
        authTokenTimestamp = Date.now(); // Record the time the token was acquired
        console.log('Authentication successful.');
    } catch (err) {
        console.error('Failed to authenticate:', err.message);
        process.exit(1);
    }

    // Iterate over patients and get transfers
    for (const patient of patients) {
        const patientIDx = patient.PatientIDx;
        console.log(`Processing PatientIDx: ${patientIDx}`);

        try {
            const transfersData = await getTransfers(patientIDx);

            if (transfersData && transfersData.TransferIDList && transfersData.TransferIDList.length > 0) {
                // Iterate over Transfer IDs and get transfer info
                for (const transferID of transfersData.TransferIDList) {
                    console.log(`  Processing TransferID: ${transferID}`);
                    try {
                        await getTransferInfo(transferID);
                    } catch (err) {
                        console.error(`Error processing TransferID ${transferID}:`, err.message);
                    }
                }
            }
        } catch (err) {
            console.error(`Error processing PatientIDx ${patientIDx}:`, err.message);
        }
    }

    console.log('All transfers and transfer info retrieved.');

    // Close the log file
    logFile.end();
})();