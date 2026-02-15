import Foundation
import Metal

struct ParamsCPU {
    var wbTemp: Float = 6500.0
    var wbTint: Float = 0.0
    var fracRx: Float = 0.7
    var fracGx: Float = 0.7
    var fracBY: Float = 1.0
    var gammaRx: Float = 1.0
    var gammaRy: Float = 1.0
    var gammaGx: Float = 1.0
    var gammaGy: Float = 1.0
    var gammaBY: Float = 1.0
    var exposure: Float = 1.0
    var width: UInt32 = 1
    var height: UInt32 = 1
}

func shaderSource() -> String {
    return """
#include <metal_stdlib>
using namespace metal;

struct Params {
    float wbTemp;
    float wbTint;
    float fracRx;
    float fracGx;
    float fracBY;
    float gammaRx;
    float gammaRy;
    float gammaGx;
    float gammaGy;
    float gammaBY;
    float exposure;
    uint width;
    uint height;
};

inline float3 kelvin_to_rgb(float kelvin) {
    kelvin = clamp(kelvin, 1000.0f, 40000.0f);
    float tmp = kelvin / 100.0f;
    float red;
    float green;
    float blue;

    if (tmp <= 66.0f) {
        red = 255.0f;
    } else {
        red = 329.698727446f * pow(tmp - 60.0f, -0.1332047592f);
    }

    if (tmp <= 66.0f) {
        green = 99.4708025861f * log(tmp) - 161.1195681661f;
    } else {
        green = 288.1221695283f * pow(tmp - 60.0f, -0.0755148492f);
    }

    if (tmp >= 66.0f) {
        blue = 255.0f;
    } else if (tmp <= 19.0f) {
        blue = 0.0f;
    } else {
        blue = 138.5177312231f * log(tmp - 10.0f) - 305.0447927307f;
    }

    return float3(clamp(red, 0.0f, 255.0f) / 255.0f,
                  clamp(green, 0.0f, 255.0f) / 255.0f,
                  clamp(blue, 0.0f, 255.0f) / 255.0f);
}

kernel void process_frame(device const packed_float3 *inBuf [[buffer(0)]],
                          device packed_float3 *outBuf [[buffer(1)]],
                          constant Params &p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    uint idx = gid.y * p.width + gid.x;

    float3 srcRGB = kelvin_to_rgb(p.wbTemp);
    float3 refRGB = kelvin_to_rgb(6500.0f);
    float3 gains = refRGB / (srcRGB + float3(1.0e-8f));

    float tintNorm = clamp(p.wbTint, -100.0f, 100.0f) / 100.0f;
    gains.y *= (1.0f - 0.15f * tintNorm);

    float3 inPx = float3(inBuf[idx]);
    float3 wb = clamp(inPx * gains, 0.0f, 1.0f);
    float Z1 = wb.x;
    float Z2 = wb.y;
    float Z3 = wb.z;

    float fracBy = max(p.fracBY, 1.0e-6f);
    float fracRx = max(p.fracRx, 1.0e-6f);
    float fracGx = max(p.fracGx, 1.0e-6f);
    float fracRy = 1.0f - fracRx;
    float fracGy = 1.0f - fracGx;

    float gammaBY = max(p.gammaBY, 1.0e-6f);
    float gammaRx = max(p.gammaRx, 1.0e-6f);
    float gammaRy = max(p.gammaRy, 1.0e-6f);
    float gammaGx = max(p.gammaGx, 1.0e-6f);
    float gammaGy = max(p.gammaGy, 1.0e-6f);

    float innerY = clamp(1.0f - (Z3 / fracBy), 0.0f, 1.0f);
    float Y = 1.0f - pow(innerY, 1.0f / gammaBY);

    float tmp1 = pow(1.0f - Y, gammaRy);
    float termR = fracRy * (1.0f - tmp1);
    float innerX1 = clamp(1.0f - ((Z1 - termR) / fracRx), 0.0f, 1.0f);
    float X1 = 1.0f - pow(innerX1, 1.0f / gammaRx);

    float tmp2 = pow(1.0f - Y, gammaGy);
    float termG = fracGy * (1.0f - tmp2);
    float innerX2 = clamp(1.0f - ((Z2 - termG) / fracGx), 0.0f, 1.0f);
    float X2 = 1.0f - pow(innerX2, 1.0f / gammaGx);

    float3 outVal = clamp(float3(Y, X1, X2) * p.exposure, 0.0f, 1.0f);
    outBuf[idx] = packed_float3(outVal);
}
"""
}

func asFloat(_ any: Any?, default defaultValue: Float) -> Float {
    guard let any = any else { return defaultValue }
    if let f = any as? Float { return f }
    if let d = any as? Double { return Float(d) }
    if let i = any as? Int { return Float(i) }
    if let s = any as? String, let d = Double(s) { return Float(d) }
    return defaultValue
}

func asInt(_ any: Any?, default defaultValue: Int) -> Int {
    guard let any = any else { return defaultValue }
    if let i = any as? Int { return i }
    if let u = any as? UInt32 { return Int(u) }
    if let d = any as? Double { return Int(d) }
    if let s = any as? String, let i = Int(s) { return i }
    return defaultValue
}

func readExact(_ count: Int) -> Data? {
    if count <= 0 {
        return Data()
    }
    var out = Data()
    while out.count < count {
        let chunk = FileHandle.standardInput.readData(ofLength: count - out.count)
        if chunk.isEmpty {
            return nil
        }
        out.append(chunk)
    }
    return out
}

func writeMessage(_ header: [String: Any], payload: Data = Data()) {
    var hdr = header
    hdr["payload_bytes"] = payload.count
    let json = (try? JSONSerialization.data(withJSONObject: hdr, options: [])) ?? Data("{}".utf8)
    var lenLE = UInt32(json.count).littleEndian
    let lenData = withUnsafeBytes(of: &lenLE) { Data($0) }
    FileHandle.standardOutput.write(lenData)
    FileHandle.standardOutput.write(json)
    if !payload.isEmpty {
        FileHandle.standardOutput.write(payload)
    }
}

func readMessage() -> ([String: Any], Data)? {
    guard let rawLen = readExact(4) else {
        return nil
    }
    if rawLen.count != 4 {
        return nil
    }
    let headerLen: Int = rawLen.withUnsafeBytes { raw in
        var v: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &v) { dst in
            dst.copyBytes(from: raw)
        }
        return Int(UInt32(littleEndian: v))
    }
    if headerLen <= 0 || headerLen > (16 * 1024 * 1024) {
        writeMessage(["ok": false, "error": "Invalid header length"])
        return nil
    }
    guard let headerData = readExact(headerLen) else {
        return nil
    }
    guard
        let obj = try? JSONSerialization.jsonObject(with: headerData, options: []),
        let dict = obj as? [String: Any]
    else {
        writeMessage(["ok": false, "error": "Invalid JSON header"])
        return nil
    }

    let payloadLen = max(0, asInt(dict["payload_bytes"], default: 0))
    let payload: Data
    if payloadLen > 0 {
        guard let p = readExact(payloadLen) else {
            writeMessage(["ok": false, "error": "Unexpected EOF reading payload"])
            return nil
        }
        payload = p
    } else {
        payload = Data()
    }

    return (dict, payload)
}

final class MetalProcessor {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var inBuffer: MTLBuffer?
    private var outBuffer: MTLBuffer?
    private var paramsBuffer: MTLBuffer?
    private var capacityBytes: Int = 0

    init() throws {
        guard let d = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "metal_worker", code: 1, userInfo: [NSLocalizedDescriptionKey: "No Metal device available"])
        }
        guard let q = d.makeCommandQueue() else {
            throw NSError(domain: "metal_worker", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }

        let library: MTLLibrary
        do {
            library = try d.makeLibrary(source: shaderSource(), options: nil)
        } catch {
            throw NSError(domain: "metal_worker", code: 3, userInfo: [NSLocalizedDescriptionKey: "Shader compile failed: \(error)"])
        }

        guard let fn = library.makeFunction(name: "process_frame") else {
            throw NSError(domain: "metal_worker", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing process_frame kernel"])
        }

        do {
            pipeline = try d.makeComputePipelineState(function: fn)
        } catch {
            throw NSError(domain: "metal_worker", code: 5, userInfo: [NSLocalizedDescriptionKey: "Pipeline build failed: \(error)"])
        }

        device = d
        queue = q
    }

    private func ensureBuffers(minBytes: Int) throws {
        if minBytes <= capacityBytes, inBuffer != nil, outBuffer != nil, paramsBuffer != nil {
            return
        }
        guard let inBuf = device.makeBuffer(length: minBytes, options: .storageModeShared) else {
            throw NSError(domain: "metal_worker", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate input buffer"])
        }
        guard let outBuf = device.makeBuffer(length: minBytes, options: .storageModeShared) else {
            throw NSError(domain: "metal_worker", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate output buffer"])
        }
        guard let pBuf = device.makeBuffer(length: MemoryLayout<ParamsCPU>.stride, options: .storageModeShared) else {
            throw NSError(domain: "metal_worker", code: 8, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate params buffer"])
        }
        inBuffer = inBuf
        outBuffer = outBuf
        paramsBuffer = pBuf
        capacityBytes = minBytes
    }

    func process(payload: Data, params: ParamsCPU) throws -> Data {
        let bytesCount = payload.count
        try ensureBuffers(minBytes: bytesCount)
        guard let inBuffer = inBuffer, let outBuffer = outBuffer, let paramsBuffer = paramsBuffer else {
            throw NSError(domain: "metal_worker", code: 12, userInfo: [NSLocalizedDescriptionKey: "Internal buffer state invalid"])
        }

        payload.copyBytes(to: UnsafeMutableRawBufferPointer(start: inBuffer.contents(), count: bytesCount))

        var localParams = params
        withUnsafePointer(to: &localParams) { ptr in
            paramsBuffer.contents().copyMemory(from: ptr, byteCount: MemoryLayout<ParamsCPU>.stride)
        }

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw NSError(domain: "metal_worker", code: 9, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"])
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw NSError(domain: "metal_worker", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to create command encoder"])
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let w = Int(params.width)
        let h = Int(params.height)
        let threadsW = max(1, pipeline.threadExecutionWidth)
        let threadsH = max(1, pipeline.maxTotalThreadsPerThreadgroup / threadsW)
        let threadsPerGroup = MTLSize(width: threadsW, height: threadsH, depth: 1)
        let threadgroups = MTLSize(
            width: (w + threadsW - 1) / threadsW,
            height: (h + threadsH - 1) / threadsH,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw NSError(domain: "metal_worker", code: 11, userInfo: [NSLocalizedDescriptionKey: "Command buffer failed: \(error)"])
        }

        return Data(bytes: outBuffer.contents(), count: bytesCount)
    }
}

var processor: MetalProcessor?
var params = ParamsCPU()
var width = 1
var height = 1

while true {
    autoreleasepool {
        guard let (header, payload) = readMessage() else {
            exit(0)
        }
        let cmd = (header["cmd"] as? String ?? "").lowercased()

        do {
            switch cmd {
            case "init":
                width = max(1, asInt(header["width"], default: width))
                height = max(1, asInt(header["height"], default: height))
                if processor == nil {
                    processor = try MetalProcessor()
                }
                params.width = UInt32(width)
                params.height = UInt32(height)
                writeMessage(["ok": true, "cmd": "init"])

            case "set_params":
                params.wbTemp = asFloat(header["wb_temp"], default: params.wbTemp)
                params.wbTint = asFloat(header["wb_tint"], default: params.wbTint)
                params.fracRx = asFloat(header["fracRx"], default: params.fracRx)
                params.fracGx = asFloat(header["fracGx"], default: params.fracGx)
                params.fracBY = asFloat(header["fracBY"], default: params.fracBY)
                params.gammaRx = asFloat(header["gammaRx"], default: params.gammaRx)
                params.gammaRy = asFloat(header["gammaRy"], default: params.gammaRy)
                params.gammaGx = asFloat(header["gammaGx"], default: params.gammaGx)
                params.gammaGy = asFloat(header["gammaGy"], default: params.gammaGy)
                params.gammaBY = asFloat(header["gammaBY"], default: params.gammaBY)
                params.exposure = asFloat(header["exposure"], default: params.exposure)
                writeMessage(["ok": true, "cmd": "set_params"])

            case "process_frame":
                guard let proc = processor else {
                    throw NSError(domain: "metal_worker", code: 20, userInfo: [NSLocalizedDescriptionKey: "Worker not initialized"])
                }
                width = max(1, asInt(header["width"], default: width))
                height = max(1, asInt(header["height"], default: height))
                params.width = UInt32(width)
                params.height = UInt32(height)

                let expected = width * height * 3 * MemoryLayout<Float>.size
                if payload.count != expected {
                    throw NSError(domain: "metal_worker", code: 21, userInfo: [NSLocalizedDescriptionKey: "Invalid frame payload size: got \(payload.count), expected \(expected)"])
                }

                let out = try proc.process(payload: payload, params: params)
                writeMessage(["ok": true, "cmd": "process_frame"], payload: out)

            case "close":
                writeMessage(["ok": true, "cmd": "close"])
                exit(0)

            default:
                throw NSError(domain: "metal_worker", code: 30, userInfo: [NSLocalizedDescriptionKey: "Unknown command: \(cmd)"])
            }
        } catch {
            writeMessage(["ok": false, "error": "\(error)"])
        }
    }
}
