// code/step2_get_treatments.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');

/** BEGIN MODIFICATIONS **/

// Ensure the output directory exists
const outputDir = path.resolve(__dirname, '../output/treatments');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Set up logging
const logFilePath = path.resolve(__dirname, '../output/step2_get_treatments.log');
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

/** END MODIFICATIONS **/

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

/** BEGIN MODIFICATIONS **/

// Declare authToken and authTokenTimestamp at the top-level scope
let authToken;
let authTokenTimestamp;

/** END MODIFICATIONS **/

// Authenticate to get the token
async function authenticate() {
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Login Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Login Request',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: 'https://192.168.1.100:4000/login?Username=web&Password=web',
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['login'],
                            query: [
                                { key: 'Username', value: 'web' },
                                { key: 'Password', value: 'web' }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', function (err, args) {
            if (err) {
                console.error('Authentication request failed:', err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const responseJson = JSON.parse(responseBody);
                    const token = responseJson.Token;
                    resolve(token);
                } else {
                    console.error('Authentication failed with status code:', response.code);
                    reject(new Error('Authentication failed'));
                }
            }
        });
    });
}

/** BEGIN MODIFICATIONS **/

// Function to check if the token has expired
async function checkTokenExpiration() {
    const currentTime = Date.now();
    if (currentTime - authTokenTimestamp >= 60 * 60 * 1000) { // 60 minutes
        console.log('Token expired. Re-authenticating...');
        try {
            authToken = await authenticate();
            authTokenTimestamp = Date.now();
            console.log('Re-authentication successful.');
        } catch (err) {
            console.error('Failed to re-authenticate:', err);
            process.exit(1);
        }
    }
}

/** END MODIFICATIONS **/

// Function to get treatments for a patient
async function getTreatments(patientIDx) {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Get Treatments Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get Treatments',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/treatment?PatientIDx=${encodeURIComponent(patientIDx)}`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'treatment'],
                            query: [
                                { key: 'PatientIDx', value: patientIDx }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error(`Request failed for PatientIDx ${patientIDx}:`, err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    const sanitizedPatientIDx = sanitizeFilename(patientIDx);
                    const filePath = path.join(outputDir, `treatments_${sanitizedPatientIDx}.json`);
                    fs.writeFile(filePath, responseBody, function (err) {
                        if (err) {
                            console.error(`Failed to save treatments for PatientIDx ${patientIDx}:`, err);
                            reject(err);
                        } else {
                            console.log(`Treatments for PatientIDx ${patientIDx} saved successfully.`);
                            resolve();
                        }
                    });
                } else if (response.code === 401) {
                    console.error('Token expired during request. Re-authenticating...');
                    try {
                        authToken = await authenticate();
                        authTokenTimestamp = Date.now();
                        console.log('Re-authentication successful. Retrying request...');
                        // Retry the request
                        await getTreatments(patientIDx);
                        resolve();
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    console.log(`No treatments found for PatientIDx ${patientIDx}.`);
                    resolve();
                } else {
                    console.error(`Error retrieving treatments for PatientIDx ${patientIDx}: Status code ${response.code}`);
                    reject(new Error(`Status code ${response.code}`));
                }
            }
        });
    });
}

/** BEGIN MODIFICATIONS **/

// Function to sanitize filenames
function sanitizeFilename(name) {
    return name.replace(/[^a-z0-9]/gi, '_');
}

// Delay function to enforce API rate limit
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/** END MODIFICATIONS **/

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

    /** BEGIN MODIFICATIONS **/

    // Set initial timestamp
    let lastRequestTime = Date.now();

    // Iterate over patients and get treatments
    for (const patient of patients) {
        const patientIDx = patient.PatientIDx;
        console.log(`Processing PatientIDx: ${patientIDx}`);

        // Enforce API rate limit of 10 requests per second
        const currentTime = Date.now();
        const timeSinceLastRequest = currentTime - lastRequestTime;
        if (timeSinceLastRequest < 100) { // 100 ms between requests = 10 requests per second
            await delay(100 - timeSinceLastRequest);
        }
        lastRequestTime = Date.now();

        try {
            await getTreatments(patientIDx);
        } catch (err) {
            console.error(`Error processing PatientIDx ${patientIDx}:`, err);
        }
    }

    console.log('All treatments retrieved.');

    // Close the log file
    logFile.end();

    /** END MODIFICATIONS **/
})();