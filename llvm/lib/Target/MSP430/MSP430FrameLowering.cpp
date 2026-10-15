//===-- MSP430FrameLowering.cpp - MSP430 Frame Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "MSP430FrameLowering.h"
#include "MSP430InstrInfo.h"
#include "MSP430MachineFunctionInfo.h"
#include "MSP430Subtarget.h"
#include "llvm/CodeGen/CFIInstBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

MSP430FrameLowering::MSP430FrameLowering(const MSP430Subtarget &STI)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(2), -2,
                          Align(2)),
      STI(STI), TII(*STI.getInstrInfo()), TRI(STI.getRegisterInfo()) {}

/// Return the number of registers MI pushes, or 0 if it is not a push.
static unsigned getNumPushedRegs(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case MSP430::PUSH16r:
    return 1;
  case MSP430::PUSHM16r:
    return MI.getOperand(0).getImm();
  default:
    return 0;
  }
}

/// Return the number of registers MI pops, or 0 if it is not a pop.
static unsigned getNumPoppedRegs(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case MSP430::POP16r:
    return 1;
  case MSP430::POPM16r:
    return MI.getOperand(0).getImm();
  default:
    return 0;
  }
}

/// Split CSI into ranges that a single PUSHM or POPM can transfer. Each
/// register in a range is one below the previous one.
///
/// Without MSP430X, every register is its own range. The callers emit PUSH and
/// POP for single registers, so this is what limits PUSHM and POPM to MSP430X.
static SmallVector<ArrayRef<CalleeSavedInfo>>
getPushPopRanges(ArrayRef<CalleeSavedInfo> CSI, const MSP430Subtarget &STI) {
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  SmallVector<ArrayRef<CalleeSavedInfo>> Ranges;
  while (!CSI.empty()) {
    size_t N = 1;
    if (STI.hasMSP430X()) {
      unsigned First = TRI->getEncodingValue(CSI.front().getReg());
      while (N < CSI.size() &&
             TRI->getEncodingValue(CSI[N].getReg()) + N == First)
        ++N;
    }
    Ranges.push_back(CSI.take_front(N));
    CSI = CSI.drop_front(N);
  }
  return Ranges;
}

bool MSP430FrameLowering::hasFPImpl(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  return (MF.disableFramePointerElim() ||
          MF.getFrameInfo().hasVarSizedObjects() || MFI.isFrameAddressTaken());
}

bool MSP430FrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo().hasVarSizedObjects();
}

void MSP430FrameLowering::BuildCFI(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL,
                                   const MCCFIInstruction &CFIInst,
                                   MachineInstr::MIFlag Flag) const {
  MachineFunction &MF = *MBB.getParent();
  unsigned CFIIndex = MF.addFrameInst(CFIInst);
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(Flag);
}

void MSP430FrameLowering::emitCalleeSavedFrameMoves(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, bool IsPrologue) const {
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const MCRegisterInfo *MRI = MF.getContext().getRegisterInfo();

  // Add callee saved registers to move list.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

  // Calculate offsets.
  for (const CalleeSavedInfo &I : CSI) {
    int64_t Offset = MFI.getObjectOffset(I.getFrameIdx());
    MCRegister Reg = I.getReg();
    unsigned DwarfReg = MRI->getDwarfRegNum(Reg, true);

    if (IsPrologue) {
      BuildCFI(MBB, MBBI, DL,
               MCCFIInstruction::createOffset(nullptr, DwarfReg, Offset));
    } else {
      BuildCFI(MBB, MBBI, DL,
               MCCFIInstruction::createRestore(nullptr, DwarfReg));
    }
  }
}

void MSP430FrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  const MSP430InstrInfo &TII =
      *static_cast<const MSP430InstrInfo *>(MF.getSubtarget().getInstrInfo());

  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI.getStackSize();
  int stackGrowth = -2;

  uint64_t NumBytes = 0;
  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - MSP430FI->getCalleeSavedFrameSize();

    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI.setOffsetAdjustment(-NumBytes);

    // Save FP into the appropriate stack slot...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::PUSH16r))
        .addReg(MSP430::R4, RegState::Kill)
        .setMIFlag(MachineInstr::FrameSetup);

    // Mark the place where FP was saved.
    // Define the current CFA rule to use the provided offset.
    BuildCFI(MBB, MBBI, DL,
             MCCFIInstruction::cfiDefCfaOffset(nullptr, -2 * stackGrowth),
             MachineInstr::FrameSetup);

    // Change the rule for the FramePtr to be an "offset" rule.
    unsigned DwarfFramePtr = TRI->getDwarfRegNum(MSP430::R4, true);
    BuildCFI(
        MBB, MBBI, DL,
        MCCFIInstruction::createOffset(nullptr, DwarfFramePtr, 2 * stackGrowth),
        MachineInstr::FrameSetup);

    // Update FP with the new base value...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::MOV16rr), MSP430::R4)
        .addReg(MSP430::SP)
        .setMIFlag(MachineInstr::FrameSetup);

    // Mark effective beginning of when frame pointer becomes valid.
    // Define the current CFA to use the FP register.
    BuildCFI(MBB, MBBI, DL,
             MCCFIInstruction::createDefCfaRegister(nullptr, DwarfFramePtr),
             MachineInstr::FrameSetup);

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineBasicBlock &MBBJ : llvm::drop_begin(MF))
      MBBJ.addLiveIn(MSP430::R4);
  } else
    NumBytes = StackSize - MSP430FI->getCalleeSavedFrameSize();

  // Skip the callee-saved push instructions.
  int StackOffset = stackGrowth;
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup) &&
         getNumPushedRegs(*MBBI)) {
    StackOffset += getNumPushedRegs(*MBBI) * stackGrowth;
    ++MBBI;

    if (!hasFP(MF)) {
      // Mark callee-saved push instruction.
      // Define the current CFA rule to use the provided offset.
      assert(StackSize && "Expected stack frame");
      BuildCFI(MBB, MBBI, DL,
               MCCFIInstruction::cfiDefCfaOffset(nullptr, -StackOffset),
               MachineInstr::FrameSetup);
    }
  }

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  if (NumBytes) { // adjust stack pointer: SP -= numbytes
    // If there is an SUB16ri of SP immediately before this instruction, merge
    // the two.
    //NumBytes -= mergeSPUpdates(MBB, MBBI, true);
    // If there is an ADD16ri or SUB16ri of SP immediately after this
    // instruction, merge the two instructions.
    // mergeSPUpdatesDown(MBB, MBBI, &NumBytes);

    if (NumBytes) {
      MachineInstr *MI =
          BuildMI(MBB, MBBI, DL, TII.get(MSP430::SUB16ri), MSP430::SP)
              .addReg(MSP430::SP)
              .addImm(NumBytes)
              .setMIFlag(MachineInstr::FrameSetup);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
    if (!hasFP(MF)) {
      // Adjust the previous CFA value if CFA was not redefined by FP
      BuildCFI(
          MBB, MBBI, DL,
          MCCFIInstruction::cfiDefCfaOffset(nullptr, StackSize - stackGrowth),
          MachineInstr::FrameSetup);
    }
  }

  emitCalleeSavedFrameMoves(MBB, MBBI, DL, true);
}

void MSP430FrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  const MSP430InstrInfo &TII =
      *static_cast<const MSP430InstrInfo *>(MF.getSubtarget().getInstrInfo());

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();

  switch (RetOpcode) {
  case MSP430::RET:
  case MSP430::RETI: break;  // These are ok
  default:
    llvm_unreachable("Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t StackSize = MFI.getStackSize();
  unsigned CSSize = MSP430FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = 0;

  MachineBasicBlock::iterator AfterPop = MBBI;
  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - CSSize;

    // pop FP.
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::POP16r), MSP430::R4)
        .setMIFlag(MachineInstr::FrameDestroy);
    unsigned DwarfStackPtr = TRI->getDwarfRegNum(MSP430::SP, true);
    BuildCFI(MBB, MBBI, DL,
             MCCFIInstruction::cfiDefCfa(nullptr, DwarfStackPtr, 2),
             MachineInstr::FrameDestroy);
    --MBBI;
    if (!MBB.succ_empty() && !MBB.isReturnBlock()) {
      unsigned DwarfFramePtr = TRI->getDwarfRegNum(MSP430::R4, true);
      BuildCFI(MBB, AfterPop, DL,
               MCCFIInstruction::createRestore(nullptr, DwarfFramePtr),
               MachineInstr::FrameDestroy);
      --MBBI;
      --AfterPop;
    }
  } else
    NumBytes = StackSize - CSSize;

  // Skip the callee-saved pop instructions.
  MachineBasicBlock::iterator FirstCSPop = MBBI;
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = std::prev(MBBI);
    if ((!getNumPoppedRegs(*PI) || !PI->getFlag(MachineInstr::FrameDestroy)) &&
        !PI->isTerminator())
      break;
    FirstCSPop = PI;
    --MBBI;
  }
  MBBI = FirstCSPop;

  DL = MBBI->getDebugLoc();

  // If there is an ADD16ri or SUB16ri of SP immediately before this
  // instruction, merge the two instructions.
  //if (NumBytes || MFI.hasVarSizedObjects())
  //  mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  if (MFI.hasVarSizedObjects()) {
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::MOV16rr), MSP430::SP)
        .addReg(MSP430::R4)
        .setMIFlag(MachineInstr::FrameDestroy);
    if (CSSize) {
      MachineInstr *MI =
          BuildMI(MBB, MBBI, DL, TII.get(MSP430::SUB16ri), MSP430::SP)
              .addReg(MSP430::SP)
              .addImm(CSSize)
              .setMIFlag(MachineInstr::FrameDestroy);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  } else {
    // adjust stack pointer back: SP += numbytes
    if (NumBytes) {
      MachineInstr *MI =
          BuildMI(MBB, MBBI, DL, TII.get(MSP430::ADD16ri), MSP430::SP)
              .addReg(MSP430::SP)
              .addImm(NumBytes)
              .setMIFlag(MachineInstr::FrameDestroy);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();

      if (!hasFP(MF)) {
        // Adjust CFA value if it was defined by SP
        BuildCFI(MBB, MBBI, DL,
                 MCCFIInstruction::cfiDefCfaOffset(nullptr, CSSize + 2),
                 MachineInstr::FrameDestroy);
      }
    }
  }

  if (!hasFP(MF)) {
    MBBI = FirstCSPop;
    int64_t Offset = -(int64_t)CSSize - 2;
    // Mark callee-saved pop instruction.
    // Define the current CFA rule to use the provided offset.
    while (MBBI != MBB.end()) {
      unsigned NumRegs = getNumPoppedRegs(*MBBI);
      ++MBBI;
      if (NumRegs) {
        Offset += 2 * NumRegs;
        BuildCFI(MBB, MBBI, DL,
                 MCCFIInstruction::cfiDefCfaOffset(nullptr, -Offset),
                 MachineInstr::FrameDestroy);
      }
    }
  }
  emitCalleeSavedFrameMoves(MBB, AfterPop, DL, false);
}

bool MSP430FrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  // The spill slots are the frame positions of the saved registers, and the
  // CFI describes them. PUSHM stores the highest register at the highest
  // address. The generic code gives the first entry the highest slot, so sort
  // the registers in descending order. Plain MSP430 keeps the default order.
  if (STI.hasMSP430X())
    llvm::sort(CSI, [TRI](const CalleeSavedInfo &A, const CalleeSavedInfo &B) {
      return TRI->getEncodingValue(A.getReg()) >
             TRI->getEncodingValue(B.getReg());
    });
  return false;
}

// FIXME: Can we eleminate these in favour of generic code?
bool MSP430FrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MSP430MachineFunctionInfo *MFI = MF.getInfo<MSP430MachineFunctionInfo>();
  MFI->setCalleeSavedFrameSize(CSI.size() * 2);

  for (ArrayRef<CalleeSavedInfo> Range : getPushPopRanges(CSI, STI)) {
    // Add the callee-saved registers as live-in. They're killed at the spill.
    for (const CalleeSavedInfo &I : Range)
      MBB.addLiveIn(I.getReg());

    // Ranges with more than one register only exist on MSP430X.
    MCRegister Highest = Range.front().getReg();
    if (Range.size() == 1) {
      BuildMI(MBB, MI, DL, TII.get(MSP430::PUSH16r))
          .addReg(Highest, RegState::Kill)
          .setMIFlag(MachineInstr::FrameSetup);
      continue;
    }

    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(MSP430::PUSHM16r))
                                  .addImm(Range.size())
                                  .addReg(Highest, RegState::Kill)
                                  .setMIFlag(MachineInstr::FrameSetup);
    for (const CalleeSavedInfo &I : Range.drop_front())
      MIB.addReg(I.getReg(), RegState::Implicit | RegState::Kill);
  }
  return true;
}

bool MSP430FrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return false;

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

  SmallVector<ArrayRef<CalleeSavedInfo>> Ranges = getPushPopRanges(CSI, STI);
  for (ArrayRef<CalleeSavedInfo> Range : llvm::reverse(Ranges)) {
    // Ranges with more than one register only exist on MSP430X.
    MCRegister Highest = Range.front().getReg();
    if (Range.size() == 1) {
      BuildMI(MBB, MI, DL, TII.get(MSP430::POP16r), Highest)
          .setMIFlag(MachineInstr::FrameDestroy);
      continue;
    }

    // The register operand of POPM16r is a use. Mark it undef and add the
    // popped registers as implicit defs.
    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(MSP430::POPM16r))
                                  .addImm(Range.size())
                                  .addReg(Highest, RegState::Undef)
                                  .setMIFlag(MachineInstr::FrameDestroy);
    for (const CalleeSavedInfo &I : Range)
      MIB.addReg(I.getReg(), RegState::ImplicitDefine);
  }

  return true;
}

MachineBasicBlock::iterator MSP430FrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const MSP430InstrInfo &TII =
      *static_cast<const MSP430InstrInfo *>(MF.getSubtarget().getInstrInfo());
  if (!hasReservedCallFrame(MF)) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub SP, <amt>' and the
    // adjcallstackdown instruction into 'add SP, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    MachineInstr &Old = *I;
    uint64_t Amount = TII.getFrameSize(Old);
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      Amount = alignTo(Amount, getStackAlign());

      MachineInstr *New = nullptr;
      if (Old.getOpcode() == TII.getCallFrameSetupOpcode()) {
        New =
            BuildMI(MF, Old.getDebugLoc(), TII.get(MSP430::SUB16ri), MSP430::SP)
                .addReg(MSP430::SP)
                .addImm(Amount);
      } else {
        assert(Old.getOpcode() == TII.getCallFrameDestroyOpcode());
        // factor out the amount the callee already popped.
        Amount -= TII.getFramePoppedByCallee(Old);
        if (Amount)
          New = BuildMI(MF, Old.getDebugLoc(), TII.get(MSP430::ADD16ri),
                        MSP430::SP)
                    .addReg(MSP430::SP)
                    .addImm(Amount);
      }

      if (New) {
        // The SRW implicit def is dead.
        New->getOperand(3).setIsDead();

        // Replace the pseudo instruction with a new instruction...
        MBB.insert(I, New);
      }
    }
  } else if (I->getOpcode() == TII.getCallFrameDestroyOpcode()) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.
    if (uint64_t CalleeAmt = TII.getFramePoppedByCallee(*I)) {
      MachineInstr &Old = *I;
      MachineInstr *New =
          BuildMI(MF, Old.getDebugLoc(), TII.get(MSP430::SUB16ri), MSP430::SP)
              .addReg(MSP430::SP)
              .addImm(CalleeAmt);
      if (!hasFP(MF)) {
        CFIInstBuilder(MBB, I, MachineInstr::NoFlags)
            .buildAdjustCFAOffset(CalleeAmt);
      }
      // The SRW implicit def is dead.
      New->getOperand(3).setIsDead();

      MBB.insert(I, New);
    }
  }

  return MBB.erase(I);
}

void
MSP430FrameLowering::processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                                         RegScavenger *) const {
  // Create a frame entry for the FP register that must be saved.
  if (hasFP(MF)) {
    int FrameIdx = MF.getFrameInfo().CreateFixedObject(2, -4, true);
    (void)FrameIdx;
    assert(FrameIdx == MF.getFrameInfo().getObjectIndexBegin() &&
           "Slot for FP register must be last in order to be found!");
  }
}
