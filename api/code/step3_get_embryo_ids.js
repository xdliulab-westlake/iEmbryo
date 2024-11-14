// code/step3_get_embryo_ids.js

const newman = require('newman');
const fs = require('fs');
const path = require('path');

/** BEGIN MODIFICATIONS **/

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

// Function to get embryo IDs for a treatment
async function getEmbryoIDs(patientIDx, treatmentName) {
    await checkTokenExpiration();
    return new Promise((resolve, reject) => {
        newman.run({
            collection: {
                info: {
                    name: 'Get Embryo IDs Collection',
                    schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json'
                },
                item: [{
                    name: 'Get Embryo IDs',
                    request: {
                        method: 'GET',
                        header: [
                            { key: 'API-token', value: authToken, type: 'text' },
                            { key: 'Content-Type', value: 'application/json', type: 'text' }
                        ],
                        url: {
                            raw: `https://192.168.1.100:4000/GET/embryoID?PatientIDx=${encodeURIComponent(patientIDx)}&TreatmentName=${encodeURIComponent(treatmentName)}`,
                            protocol: 'https',
                            host: ['192', '168', '1', '100'],
                            port: '4000',
                            path: ['GET', 'embryoID'],
                            query: [
                                { key: 'PatientIDx', value: patientIDx },
                                { key: 'TreatmentName', value: treatmentName }
                            ]
                        }
                    }
                }]
            },
            reporters: 'cli',
            insecure: true
        }).on('request', async function (err, args) {
            if (err) {
                console.error(`Request failed for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}:`, err);
                reject(err);
            } else {
                const response = args.response;
                if (response.code === 200) {
                    const responseBody = response.stream.toString('utf8');
                    // Sanitize file name to avoid illegal characters
                    const sanitizedPatientIDx = sanitizeFilename(patientIDx);
                    const sanitizedTreatmentName = sanitizeFilename(treatmentName);
                    const filePath = path.join(outputDir, `embryo_ids_${sanitizedPatientIDx}_${sanitizedTreatmentName}.json`);
                    fs.writeFile(filePath, responseBody, function (err) {
                        if (err) {
                            console.error(`Failed to save embryo IDs for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}:`, err);
                            reject(err);
                        } else {
                            console.log(`Embryo IDs for PatientIDx ${patientIDx}, TreatmentName ${treatmentName} saved successfully.`);
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
                        await getEmbryoIDs(patientIDx, treatmentName);
                        resolve();
                    } catch (authErr) {
                        console.error('Failed to re-authenticate:', authErr);
                        reject(authErr);
                    }
                } else if (response.code === 204) {
                    console.log(`No embryo IDs found for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}.`);
                    resolve();
                } else {
                    console.error(`Error retrieving embryo IDs for PatientIDx ${patientIDx}, TreatmentName ${treatmentName}: Status code ${response.code}`);
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

    /** END MODIFICATIONS **/

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

            /** BEGIN MODIFICATIONS **/

            // Enforce API rate limit of 10 requests per second
            const currentTime = Date.now();
            const timeSinceLastRequest = currentTime - lastRequestTime;
            if (timeSinceLastRequest < 100) { // 100 ms between requests = 10 requests per second
                await delay(100 - timeSinceLastRequest);
            }
            lastRequestTime = Date.now();

            /** END MODIFICATIONS **/

            try {
                await getEmbryoIDs(patientIDx, treatmentName);
            } catch (err) {
                console.error(`Error processing PatientIDx ${patientIDx}, TreatmentName ${treatmentName}:`, err);
            }
        }
    }

    console.log('All embryo IDs retrieved.');

    /** BEGIN MODIFICATIONS **/

    // Close the log file
    logFile.end();

    /** END MODIFICATIONS **/
})();