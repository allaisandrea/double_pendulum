// avprobe: lists the cameras macOS offers, or streams one and prints its mean
// brightness and frame rate twice a second. Brightness is how to confirm that
// an exposure change made with uvc-util actually reached the sensor.
//
//   avprobe                  list cameras
//   avprobe <name> [secs]    stream the first camera whose name contains <name>
import AVFoundation
import Foundation

let args = CommandLine.arguments
let match = args.count > 1 ? args[1] : ""
let secs = args.count > 2 ? (Double(args[2]) ?? 10) : 10

// External cameras are "ExternalUnknown" up to macOS 13 and "External" from
// macOS 14; asking for both finds them on either.
let types: [AVCaptureDevice.DeviceType] = [
    .builtInWideAngleCamera, .externalUnknown,
    AVCaptureDevice.DeviceType(rawValue: "AVCaptureDeviceTypeExternal"),
]
let cameras = AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: .unspecified).devices

if match.isEmpty {
    for c in cameras { print("\(c.localizedName)  [\(c.deviceType.rawValue)]  \(c.uniqueID)") }
    exit(0)
}
guard let camera = cameras.first(where: { $0.localizedName.contains(match) }) else {
    print("no camera matching \"\(match)\"; run avprobe with no arguments to list them")
    exit(1)
}

final class Probe: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let start = Date()
    var last = Date(), frames = 0, sum = 0.0

    func captureOutput(_ output: AVCaptureOutput, didOutput buffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(buffer) else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        // Plane 0 of 420v is luma, black at 16. Every 4th pixel is plenty.
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0)?.assumingMemoryBound(to: UInt8.self) else { return }
        let (w, h) = (CVPixelBufferGetWidthOfPlane(pb, 0), CVPixelBufferGetHeightOfPlane(pb, 0))
        let stride_ = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        var total = 0, count = 0
        for y in stride(from: 0, to: h, by: 4) {
            let row = base + y * stride_
            for x in stride(from: 0, to: w, by: 4) { total += Int(row[x]); count += 1 }
        }
        sum += Double(total) / Double(count)
        frames += 1
        let now = Date()
        if now.timeIntervalSince(last) >= 0.5 {
            print(String(format: "t=%5.1fs  mean luma %6.1f  %5.1f fps  (%dx%d)",
                         now.timeIntervalSince(start), sum / Double(frames),
                         Double(frames) / now.timeIntervalSince(last), w, h))
            fflush(stdout)
            (frames, sum, last) = (0, 0, now)
        }
    }
}

let session = AVCaptureSession()
session.addInput(try! AVCaptureDeviceInput(device: camera))
let output = AVCaptureVideoDataOutput()
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
let probe = Probe()
output.setSampleBufferDelegate(probe, queue: DispatchQueue(label: "avprobe"))
session.addOutput(output)
print("streaming \(camera.localizedName) for \(secs) s")
fflush(stdout)
session.startRunning()
RunLoop.main.run(until: Date(timeIntervalSinceNow: secs))
session.stopRunning()
