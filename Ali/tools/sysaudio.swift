// sysaudio.swift — CoreAudio process-tap system-audio capture.
//
// This replaces the earlier ScreenCaptureKit version because SCKit hit a
// macOS 26 TCC wall: adhoc-signed CLI binaries couldn't persist Screen
// Recording permission across rebuilds. CoreAudio's process-tap API
// (AudioHardwareCreateProcessTap, macOS 14.2+) doesn't require Screen
// Recording at all — taps sit at the driver level and only need audio
// permission (which the user already grants for the mic).
//
// The Rust-side Cluely clone "natively" proved this route works
// universally across Zoom / FaceTime / Meet / Slack / Discord; this
// Swift port mirrors their design:
//
//   1. Build a CATapDescription (global mono tap, exclude no processes)
//   2. AudioHardwareCreateProcessTap → an AudioObjectID we can reference
//   3. Wrap that tap in an aggregate device so CoreAudio will pump IO
//      callbacks containing the tapped audio
//   4. Install an IOProc, AudioDeviceStart, and in each callback
//      resample the Float32 PCM down to 16 kHz mono Int16 → stdout
//
// Build:
//   swiftc -O -framework CoreAudio -framework AudioToolbox \
//          -framework AVFoundation -framework CoreMedia \
//          -o tools/bin/sysaudio tools/sysaudio.swift
//
// Usage (standalone smoke test):
//   tools/bin/sysaudio | ffplay -f s16le -ar 16000 -ac 1 -i -

import Foundation
import CoreAudio
import AudioToolbox
import AVFoundation
import CoreMedia

let TARGET_SAMPLE_RATE: Double = 16000
let TARGET_CHANNELS: AVAudioChannelCount = 1

func logErr(_ s: String) {
    if let data = (s + "\n").data(using: .utf8) {
        FileHandle.standardError.write(data)
    }
}

// MARK: - CoreAudio helpers

/// Read a CFString property from an AudioObject. Returns nil on failure.
func getStringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size) == noErr else { return nil }
    var cfstr: CFString = "" as CFString
    let status = withUnsafeMutablePointer(to: &cfstr) { ptr -> OSStatus in
        AudioObjectGetPropertyData(object, &address, 0, nil, &size, ptr)
    }
    guard status == noErr else { return nil }
    return cfstr as String
}

/// Default system output device's UID — used as the aggregate device's
/// main sub-device so CoreAudio knows what clock to follow.
func defaultOutputDeviceUID() throws -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var deviceID: AudioDeviceID = 0
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID)
    guard status == noErr else {
        throw NSError(domain: "sysaudio", code: Int(status),
                      userInfo: [NSLocalizedDescriptionKey: "no default output device"])
    }
    guard let uid = getStringProperty(deviceID, kAudioDevicePropertyDeviceUID) else {
        throw NSError(domain: "sysaudio", code: -1,
                      userInfo: [NSLocalizedDescriptionKey: "output device has no UID"])
    }
    return uid
}

// MARK: - The capture pipeline

final class ProcessTapCapture {
    var tapObject: AudioObjectID = 0
    var aggregateDeviceID: AudioObjectID = 0
    var ioProcID: AudioDeviceIOProcID?
    var tapUID: String = ""
    var sourceASBD: AudioStreamBasicDescription = AudioStreamBasicDescription()
    var converter: AVAudioConverter?
    var sourceFormat: AVAudioFormat?
    var targetFormat: AVAudioFormat?
    let stdout = FileHandle.standardOutput
    let ioQueue = DispatchQueue(label: "com.ali.sysaudio.io", qos: .userInitiated)

    func start() throws {
        // 1. Create CATapDescription: global mono mixdown, exclude no
        //    processes (i.e. capture every app's audio output).
        //    CATapDescription is NSObject; its initialiser names use the
        //    mono/stereo + mixdown/global variants. We want a mono
        //    mixdown of the GLOBAL stream, excluding nothing.
        let tapDesc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        tapDesc.name = "AliSystemAudioTap"
        tapDesc.isPrivate = true
        // muteBehavior default = "unmuted" — we want to SNIFF audio
        // without silencing it for the user.

        var tapID: AudioObjectID = 0
        var status = AudioHardwareCreateProcessTap(tapDesc, &tapID)
        guard status == noErr else {
            throw NSError(
                domain: "sysaudio", code: Int(status),
                userInfo: [NSLocalizedDescriptionKey:
                    "AudioHardwareCreateProcessTap failed (\(status)). macOS 14.2+ required."])
        }
        self.tapObject = tapID

        // 2. Read back the tap's UID — this is what we plug into the
        //    aggregate device's sub-tap list.
        guard let uid = getStringProperty(tapID, kAudioTapPropertyUID) else {
            throw NSError(domain: "sysaudio", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "tap created but has no UID"])
        }
        self.tapUID = uid
        logErr("sysaudio: tap created uid=\(uid)")

        // 3. Build an aggregate-device descriptor. The tap becomes a
        //    sub-tap; the default output device becomes the main
        //    sub-device so the aggregate inherits its clock.
        let outputUID = try defaultOutputDeviceUID()
        let aggUID = UUID().uuidString

        let subTap: [String: Any] = [
            kAudioSubTapUIDKey as String: uid
        ]
        let subDevice: [String: Any] = [
            kAudioSubDeviceUIDKey as String: outputUID
        ]

        let aggDesc: [String: Any] = [
            kAudioAggregateDeviceUIDKey as String: aggUID,
            kAudioAggregateDeviceNameKey as String: "AliSystemAudioAgg",
            kAudioAggregateDeviceIsPrivateKey as String: 1,
            kAudioAggregateDeviceIsStackedKey as String: 0,
            kAudioAggregateDeviceTapAutoStartKey as String: 1,
            kAudioAggregateDeviceMainSubDeviceKey as String: outputUID,
            kAudioAggregateDeviceSubDeviceListKey as String: [subDevice],
            kAudioAggregateDeviceTapListKey as String: [subTap],
        ]

        var aggID: AudioDeviceID = 0
        status = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
        guard status == noErr else {
            throw NSError(
                domain: "sysaudio", code: Int(status),
                userInfo: [NSLocalizedDescriptionKey:
                    "AudioHardwareCreateAggregateDevice failed (\(status))"])
        }
        self.aggregateDeviceID = aggID
        logErr("sysaudio: aggregate device id=\(aggID)")

        // 4. Find the tap's stream format on this aggregate device so we
        //    can set up an AVAudioConverter that targets 16 kHz mono
        //    Float32.
        var asbdAddr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        status = AudioObjectGetPropertyData(tapID, &asbdAddr, 0, nil, &asbdSize, &sourceASBD)
        if status != noErr {
            logErr("sysaudio: kAudioTapPropertyFormat failed (\(status)); falling back to default stream format")
            // Fall back: assume 48 kHz 2-channel Float32 non-interleaved
            sourceASBD = AudioStreamBasicDescription(
                mSampleRate: 48000,
                mFormatID: kAudioFormatLinearPCM,
                mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
                mBytesPerPacket: 4,
                mFramesPerPacket: 1,
                mBytesPerFrame: 4,
                mChannelsPerFrame: 2,
                mBitsPerChannel: 32,
                mReserved: 0
            )
        }
        logErr("sysaudio: source format sr=\(sourceASBD.mSampleRate) ch=\(sourceASBD.mChannelsPerFrame) flags=\(sourceASBD.mFormatFlags)")

        guard let srcFmt = AVAudioFormat(streamDescription: &sourceASBD),
              let dstFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: TARGET_SAMPLE_RATE,
                channels: TARGET_CHANNELS,
                interleaved: false
              ) else {
            throw NSError(domain: "sysaudio", code: -3,
                          userInfo: [NSLocalizedDescriptionKey: "failed to construct AVAudioFormat"])
        }
        self.sourceFormat = srcFmt
        self.targetFormat = dstFmt
        self.converter = AVAudioConverter(from: srcFmt, to: dstFmt)
        guard self.converter != nil else {
            throw NSError(domain: "sysaudio", code: -4,
                          userInfo: [NSLocalizedDescriptionKey: "AVAudioConverter init failed"])
        }

        // 5. Install IOProc and start the aggregate device.
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        var createdProcID: AudioDeviceIOProcID?
        status = AudioDeviceCreateIOProcID(aggID, ioProcTrampoline, selfPtr, &createdProcID)
        guard status == noErr, let procID = createdProcID else {
            throw NSError(domain: "sysaudio", code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "AudioDeviceCreateIOProcID failed"])
        }
        self.ioProcID = procID

        status = AudioDeviceStart(aggID, procID)
        guard status == noErr else {
            throw NSError(domain: "sysaudio", code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "AudioDeviceStart failed"])
        }
        logErr("sysaudio: capture started")
    }

    func stop() {
        if aggregateDeviceID != 0, let procID = ioProcID {
            AudioDeviceStop(aggregateDeviceID, procID)
            AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
        }
        if aggregateDeviceID != 0 {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
        }
        if tapObject != 0 {
            AudioHardwareDestroyProcessTap(tapObject)
        }
    }

    /// Called on the CoreAudio IO thread once per audio buffer.
    func handleAudio(_ inputData: UnsafePointer<AudioBufferList>) {
        guard let srcFmt = sourceFormat,
              let dstFmt = targetFormat,
              let conv = converter else { return }

        let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
        let firstBuf = abl[0]
        let bytesPerFrame = max(Int(sourceASBD.mBytesPerFrame), 1)
        let frameCount = Int(firstBuf.mDataByteSize) / bytesPerFrame
        guard frameCount > 0 else { return }

        // Wrap the raw pointer in an AVAudioPCMBuffer without copying.
        guard let srcBuf = AVAudioPCMBuffer(
            pcmFormat: srcFmt,
            bufferListNoCopy: inputData,
            deallocator: nil
        ) else { return }

        // Output capacity: sample-rate ratio plus headroom.
        let ratio = dstFmt.sampleRate / srcFmt.sampleRate
        let outCap = AVAudioFrameCount(Double(srcBuf.frameLength) * ratio) + 512
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: outCap) else { return }

        var supplied = false
        var err: NSError?
        conv.convert(to: outBuf, error: &err) { _, outStatus in
            if supplied {
                outStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return srcBuf
        }
        if let err = err {
            logErr("sysaudio: converter error: \(err)")
            return
        }

        // Pack Float32 mono → Int16 LE and write to stdout on the IO
        // queue so we don't stall CoreAudio.
        guard let floatChan = outBuf.floatChannelData else { return }
        let n = Int(outBuf.frameLength)
        let src = floatChan[0]
        var out = Data(count: n * MemoryLayout<Int16>.size)
        out.withUnsafeMutableBytes { rawBuf in
            let int16Ptr = rawBuf.bindMemory(to: Int16.self).baseAddress!
            for i in 0 ..< n {
                let clamped = max(-1.0, min(1.0, src[i]))
                int16Ptr[i] = Int16(clamped * 32767.0)
            }
        }
        ioQueue.async { [stdout] in
            stdout.write(out)
        }
    }
}

// Trampoline from C callback → Swift instance method. Signature comes
// from AudioDeviceIOProc in CoreAudio.
let ioProcTrampoline: AudioDeviceIOProc = { (
    _ inDevice: AudioObjectID,
    _ inNow: UnsafePointer<AudioTimeStamp>,
    _ inInputData: UnsafePointer<AudioBufferList>,
    _ inInputTime: UnsafePointer<AudioTimeStamp>,
    _ outOutputData: UnsafeMutablePointer<AudioBufferList>,
    _ inOutputTime: UnsafePointer<AudioTimeStamp>,
    _ inClientData: UnsafeMutableRawPointer?
) -> OSStatus in
    guard let ptr = inClientData else { return noErr }
    let capture = Unmanaged<ProcessTapCapture>.fromOpaque(ptr).takeUnretainedValue()
    capture.handleAudio(inInputData)
    return noErr
}

// MARK: - Entry

if #available(macOS 14.2, *) {
    let capture = ProcessTapCapture()
    do {
        try capture.start()
    } catch {
        logErr("sysaudio: start failed: \(error)")
        exit(5)
    }
    signal(SIGINT)  { _ in exit(0) }
    signal(SIGTERM) { _ in exit(0) }
    // IOProc callbacks run off the RunLoop, so we just park the main
    // thread here to keep the process alive.
    RunLoop.current.run()
} else {
    logErr("sysaudio: CoreAudio process tap requires macOS 14.2+")
    exit(6)
}
