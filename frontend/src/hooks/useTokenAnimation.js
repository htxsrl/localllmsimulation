import { useState, useEffect, useRef, useCallback } from 'react';

const DEMO_TEXT = `Quality, as Pirsig understood, is not a property of objects but the primordial reality from which mind and matter emerge. Federico Faggin, father of the microprocessor, now explores how qualia—the felt experience of consciousness—cannot arise from purely computational processes. Quantum physics reveals a universe where observation and reality intertwine, where particles exist in superposition until witnessed. The Lakota knew this: Mitakuye Oyasin, "we are all related," speaks to the interconnected web of existence. Perhaps consciousness is not produced by the brain but filtered through it, like sunlight through a prism. In this view, silicon cannot feel, but the universe itself may be aware—a cosmic mind dreaming matter into being, where Quality flows like a river through all things, and every qualia is a window into the infinite.`;

export function useTokenAnimation(tps, ttftMs, isActive, restartKey) {
    const [phase, setPhase] = useState('idle');
    const [displayed, setDisplayed] = useState('');
    const [ttftRemaining, setTtftRemaining] = useState(0);
    const timeoutRef = useRef(null);
    const intervalRef = useRef(null);
    const countdownRef = useRef(null);

    const stop = useCallback(() => {
        clearTimeout(timeoutRef.current);
        clearInterval(intervalRef.current);
        clearInterval(countdownRef.current);
        setPhase('stopped');
    }, []);

    useEffect(() => {
        // Clear any existing timers first
        clearTimeout(timeoutRef.current);
        clearInterval(intervalRef.current);
        clearInterval(countdownRef.current);

        if (!isActive || !tps) {
            setPhase('idle');
            setDisplayed('');
            setTtftRemaining(0);
            return;
        }

        const ttft = ttftMs || 500;
        setPhase('ttft');
        setDisplayed('');
        setTtftRemaining(ttft);

        // Countdown during TTFT
        const countdownStart = Date.now();
        countdownRef.current = setInterval(() => {
            const elapsed = Date.now() - countdownStart;
            const remaining = Math.max(0, ttft - elapsed);
            setTtftRemaining(remaining);
        }, 100);

        timeoutRef.current = setTimeout(() => {
            clearInterval(countdownRef.current);
            setTtftRemaining(0);
            setPhase('streaming');

            const charsPerToken = 4;
            const msPerChar = 1000 / (tps * charsPerToken);
            let i = 0;

            intervalRef.current = setInterval(() => {
                if (i < DEMO_TEXT.length) {
                    setDisplayed(DEMO_TEXT.slice(0, i + 1));
                    i++;
                } else {
                    setPhase('done');
                    clearInterval(intervalRef.current);
                }
            }, Math.max(10, msPerChar));
        }, ttft);

        return () => {
            clearTimeout(timeoutRef.current);
            clearInterval(intervalRef.current);
            clearInterval(countdownRef.current);
        };
    }, [tps, ttftMs, isActive, restartKey]);

    return { phase, displayed, ttftRemaining, stop };
}
