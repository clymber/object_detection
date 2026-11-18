import CoreML

/// An immutable row-major FP32 tensor that can cross detector actor boundaries.
nonisolated struct FloatTensor: Sendable {
    let shape: [Int]
    let values: [Float]

    /// Checks dimensions and storage size without silently reshaping data.
    init(shape: [Int], values: [Float]) throws {
        var count = 1
        guard !shape.isEmpty else { throw DetectorError.invalidTensor("empty shape") }
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !product.overflow else {
                throw DetectorError.invalidTensor("nonpositive or overflowing shape")
            }
            count = product.partialValue
        }
        guard values.count == count else {
            throw DetectorError.invalidTensor("shape and element count differ")
        }
        self.shape = shape
        self.values = values
    }

    /// Reads FP16/FP32 Core ML storage using its actual strides.
    init(_ array: MLMultiArray) throws {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var values = [Float](repeating: 0, count: array.count)

        /// Converts a row-major element index into a strided storage offset.
        func offset(_ index: Int) -> Int {
            var remainder = index
            var result = 0
            for axis in shape.indices.reversed() {
                result += (remainder % shape[axis]) * strides[axis]
                remainder /= shape[axis]
            }
            return result
        }

        switch array.dataType {
        case .float16:
            array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
                for index in values.indices {
                    values[index] = Float(buffer[offset(index)])
                }
            }
        case .float32:
            array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                for index in values.indices { values[index] = buffer[offset(index)] }
            }
        default:
            throw DetectorError.invalidTensor("expected FP16 or FP32 storage")
        }
        try self.init(shape: shape, values: values)
    }

    /// Creates the FP32 input multiarray, respecting Core ML's allocated strides.
    func multiArray() throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
            for index in values.indices {
                var remainder = index
                var offset = 0
                for axis in shape.indices.reversed() {
                    offset += (remainder % shape[axis]) * strides[axis]
                    remainder /= shape[axis]
                }
                buffer[offset] = values[index]
            }
        }
        return array
    }
}
