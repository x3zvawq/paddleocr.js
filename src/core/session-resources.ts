import type { OrtInferenceSession, OrtModule } from "../types/ort.ts";

/** Tracks only resources created by a pipeline factory, never injected services. */
export class SessionResources {
    private readonly disposers: (() => Promise<void>)[] = [];
    private releasePromise?: Promise<void>;

    assertActive(): void {
        if (this.releasePromise) throw new Error("Pipeline has been destroyed.");
    }

    async createSession(ort: OrtModule, modelBuffer: ArrayBuffer): Promise<OrtInferenceSession> {
        this.assertActive();
        const session = await ort.InferenceSession.create(modelBuffer);
        this.disposers.push(async () => {
            await session.release?.();
        });
        return session;
    }

    ownPipeline(pipeline: { destroy(): Promise<void> }): void {
        this.disposers.push(() => pipeline.destroy());
    }

    release(): Promise<void> {
        this.releasePromise ??= Promise.allSettled(
            this.disposers.splice(0).map(async (dispose) => dispose())
        ).then((results) => {
            const errors = results.flatMap((result) =>
                result.status === "rejected" ? [result.reason] : []
            );
            if (errors.length)
                throw new AggregateError(errors, "Failed to release model resources.");
        });
        return this.releasePromise;
    }

    async rollback(error: unknown): Promise<never> {
        try {
            await this.release();
        } catch (cleanupError) {
            throw new AggregateError(
                [error, cleanupError],
                "Model initialization and cleanup failed."
            );
        }
        throw error;
    }
}
