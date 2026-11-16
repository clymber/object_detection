import Foundation

/// Exercises the demo controller independently of SwiftUI and device signing.
@main
@MainActor
struct DemoSessionSmokeTests {
    /// Runs the lifecycle checks with short, bounded demo durations.
    static func main() async throws {
        testInitialState()
        try await testCompletion()
        try await testStop()
        try await testRestartAndDuplicateStart()
        try await testErrorAndRetry()
        print("DemoSession: all 5 lifecycle checks passed.")
    }

    /// Confirms that a new session is idle and stopping it has no effect.
    private static func testInitialState() {
        let session = DemoSession()
        session.stop()
        precondition(session.status == .idle)
        precondition(!session.isRunning)
    }

    /// Checks that Start immediately runs and eventually completes the demo.
    private static func testCompletion() async throws {
        let session = DemoSession(completionDelay: .milliseconds(10))
        session.start(model: "RF-DETR Small", input: "Static image")
        precondition(session.status == .running)
        precondition(session.message.contains("RF-DETR Small"))
        try await waitForFinish(session)
        precondition(session.status == .completed)
    }

    /// Ensures a cancelled timer cannot change Idle back to Completed later.
    private static func testStop() async throws {
        let session = DemoSession(completionDelay: .milliseconds(10))
        session.start(model: "RF-DETR Nano", input: "Prerecorded video")
        session.stop()
        precondition(session.status == .idle)
        let stoppedMessage = session.message
        try await Task.sleep(for: .milliseconds(30))
        precondition(session.status == .idle)
        precondition(session.message == stoppedMessage)
    }

    /// Checks rapid restart and ignores extra Start calls during a run.
    private static func testRestartAndDuplicateStart() async throws {
        let session = DemoSession(completionDelay: .milliseconds(10))
        session.start(model: "Old run", input: "Static image", simulateError: true)
        session.stop()
        session.start(model: "YOLOX Nano", input: "Live camera")
        let runningMessage = session.message
        session.start(model: "Extra run", input: "Static image", simulateError: true)
        precondition(session.message == runningMessage)
        try await waitForFinish(session)
        precondition(session.status == .completed)
    }

    /// Confirms the simulated error is visible and a subsequent run can recover.
    private static func testErrorAndRetry() async throws {
        let session = DemoSession(completionDelay: .milliseconds(10))
        session.start(model: "YOLOX Nano", input: "Static image", simulateError: true)
        try await waitForFinish(session)
        precondition(session.status == .error)
        precondition(session.message.contains("Simulated error"))
        session.stop()
        precondition(session.status == .error)
        session.start(model: "YOLOX Nano", input: "Static image")
        try await waitForFinish(session)
        precondition(session.status == .completed)
    }

    /// Waits for a run to finish, failing instead of hanging on a broken timer.
    private static func waitForFinish(_ session: DemoSession) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while session.isRunning && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        precondition(!session.isRunning, "Demo did not finish within two seconds.")
    }
}
